from __future__ import annotations

import json
import logging
import os
from threading import Thread
from typing import TYPE_CHECKING, Optional, Type

import requests

from .structs import Scorecard

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger()


class Swarm:
    """Orchestration for many agents playing many ARC-AGI-3 games."""

    GAMES: list[str]
    ROOT_URL: str
    COUNT: int
    agent_name: str
    agent_class: Type[Agent]
    threads: list[Thread]
    agents: list[Agent]
    record_games: list[str]
    cleanup_threads: list[Thread]
    headers: dict[str, str]
    card_id: Optional[str]
    _session: requests.Session

    def __init__(
        self,
        agent: str,
        ROOT_URL: str,
        games: list[str],
        tags: list[str] = [],
    ) -> None:
        from . import AVAILABLE_AGENTS

        self.GAMES = games
        self.ROOT_URL = ROOT_URL
        self.agent_name = agent
        self.agent_class = AVAILABLE_AGENTS[agent]
        self.threads = []
        self.agents = []
        self.cleanup_threads = []
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",  # Explicit UTF-8 for Windows - Claude Sonnet 4
        }
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        self.tags = tags.copy() if tags else []

        # Set up base tags for tracing
        if self.agent_name.endswith(".recording.jsonl"):
            # Extract GUID from playback filename
            # Format: game.agent.count.guid.recording.jsonl
            parts = self.agent_name.split(".")
            guid = parts[-3] if len(parts) >= 4 else "unknown"
            self.tags.extend(["playback", guid])
        else:
            self.tags.extend(["agent", self.agent_name])

    def main(self) -> Scorecard:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        self.card_id = self.open_scorecard()

        # create all the agents
        for i in range(len(self.GAMES)):
            g = self.GAMES[i % len(self.GAMES)]
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                cookies=self._session.cookies,
                tags=self.tags,
            )
            self.agents.append(a)

        # create all the threads
        for a in self.agents:
            self.threads.append(Thread(target=a.main, daemon=True))

        # start all the threads
        for t in self.threads:
            t.start()

        # wait for all agent to finish
        for t in self.threads:
            t.join()

        # all agents are now done
        card_id_for_url = self.card_id
        logger.debug(f"Swarm: Closing scorecard with card_id: {self.card_id}")
        scorecard = self.close_scorecard(self.card_id)
        logger.info("--- FINAL SCORECARD REPORT ---")
        logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Provide web link to scorecard
        if card_id_for_url:
            scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id_for_url}"
            logger.info(f"View your scorecard online: {scorecard_url}")

        logger.debug(f"Swarm: Starting cleanup for {len(self.agents)} agents with scorecard")
        self.cleanup(scorecard)

        return scorecard

    def open_scorecard(self) -> str:
        try:
            r = self._session.post(
                f"{self.ROOT_URL}/api/scorecard/open",
                json={"tags": self.tags},
                headers=self.headers,
                timeout=5
            )
            r.raise_for_status()  # Raise exception for HTTP errors
            response = r.json()
            
            if "error" in response:
                logger.error(f"Error opening scorecard: {response}")
                raise Exception(f"Scorecard API error: {response['error']}")
                
            card_id = str(response["card_id"])
            logger.info(f"Successfully opened scorecard with ID: {card_id}")
            return card_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to open scorecard: {str(e)}")
            raise Exception(f"Failed to open scorecard: {str(e)}")

    def close_scorecard(self, card_id: str) -> Scorecard:
        try:
            self.card_id = None
            # Ensure proper UTF-8 encoding for Windows
            data = {"card_id": card_id}
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            r = self._session.post(
                f"{self.ROOT_URL}/api/scorecard/close",
                data=json_data,
                headers={
                    **self.headers,
                    "Content-Type": "application/json; charset=utf-8"
                },
                timeout=5
            )
            r.raise_for_status()  # Raise exception for HTTP errors
            response = r.json()
            
            if "error" in response:
                logger.error(f"Error closing scorecard: {response}")
                raise Exception(f"Scorecard API error: {response['error']}")
                
            scorecard = Scorecard.model_validate(response)
            logger.info(f"Successfully closed scorecard with ID: {card_id}")
            return scorecard
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to close scorecard: {str(e)}")
            raise Exception(f"Failed to close scorecard: {str(e)}")

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Cleanup all agents and ensure proper session closure."""
        try:
            logger.debug(f"Swarm cleanup: Processing {len(self.agents)} agents")
            for i, a in enumerate(self.agents):
                logger.debug(f"Swarm cleanup: Cleaning up agent {i+1}/{len(self.agents)}: {a.name}")
                try:
                    a.cleanup(scorecard)
                except Exception as e:
                    logger.error(f"Error cleaning up agent {a.name}: {str(e)}")
            
            if hasattr(self, "_session"):
                try:
                    self._session.close()
                    logger.debug("Swarm cleanup: Session closed successfully")
                except Exception as e:
                    logger.error(f"Error closing session: {str(e)}")
            
            logger.debug("Swarm cleanup: Complete")
            
        except Exception as e:
            logger.error(f"Error during swarm cleanup: {str(e)}")
            raise
