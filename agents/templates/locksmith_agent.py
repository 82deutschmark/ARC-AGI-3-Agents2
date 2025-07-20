import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from openai import OpenAI as OpenAIClient

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
# Claude Sonnet 4: Changed from LLM to ReasoningLLM to enable reasoning capture
from .llm_agents import ReasoningLLM

logger = logging.getLogger()


# Claude Sonnet 4: Changed inheritance from LLM to ReasoningLLM to capture reasoning metadata
class LockSmithExpert(ReasoningLLM):
    """A highly specialized agent for the LockSmith game with accurate game mechanics."""
    
    MAX_ACTIONS = 10
    DO_OBSERVATION = True
    MODEL = "o4-mini-2025-04-16"
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 25
    # Claude Sonnet 4: Added REASONING_EFFORT to enable reasoning token capture for o4-mini model
    REASONING_EFFORT = "high"
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Claude Sonnet 4: Initialize reasoning-related variables to ensure proper recording
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata for LockSmith gameplay."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field - Claude Sonnet 4
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "locksmith_expert",
            "game_specialization": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o4-mini model."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings) - Claude Sonnet 4
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens
    
    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(f"""
# LOCKSMITH GAME - COMPLETE STRATEGY GUIDE
IMPORTANT: Use only standard ASCII characters in your reasoning. 
Avoid special punctuation like en-dashes, em-dashes, or smart quotes. Use regular hyphens (-) and straight quotes (\") only.
## OBJECTIVE
Modify your key to match the door's requirements, then reach the door to complete the level.

## ENERGY SYSTEM (SURVIVAL CRITICAL!)
- **Energy Display**: Purple squares at TOP of screen show remaining energy (UI indicator only)
- **Energy Loss**: Each movement consumes energy
- **Energy Refill**: Purple squares scattered in the play grid refill energy to FULL
- **Game Over**: Zero energy = death
- **Planning**: Budget extra energy for multiple changer attempts!

## KEY MODIFICATION SYSTEM (CORE MECHANIC)

### Current Key Display
- **Location**: Bottom-left corner shows your current key
- **Components**: Has both SHAPE and COLOR elements
- **Target**: Must match EXACTLY what's shown inside the door

### Shape Changers
- **Appearance**: 3 touching WHITE squares in the grid with only one other color square in the grid
- **Function**: Moving over this area changes your key's SHAPE (not color)
- **Cycling**: Each visit cycles to the next shape variation
- **Retry Method**: Move off the changer, then back on to cycle again
- **Strategy**: May need multiple attempts to get the right shape

### Color Changers  
- **Appearance**: 3x3 grid area that includes ORANGE squares at the bottom and right side
- **Function**: Moving over this area changes your key's COLOR (not shape)
- **Cycling**: Each visit cycles to the next color variation
- **Retry Method**: Move off the changer, then back on to cycle again
- **Strategy**: May need multiple attempts to get the right color

### The Door (Victory Condition)
- **Appearance**: Black square with colored shape inside
- **Target Pattern**: The shape/color inside shows exactly what your key must match
- **Victory**: Move onto door ONLY when your key matches perfectly
- **Result**: Scores a point and advances to next level

## STRATEGIC PLANNING FRAMEWORK

### 1. SITUATION ASSESSMENT
- **Energy Status**: Count purple squares at top - how much energy remains?
- **Current Key**: What shape and color is your key right now?
- **Target Key**: What does the door require (shape AND color)?
- **Gap Analysis**: What needs to change - shape, color, or both?

### 2. RESOURCE MAPPING
- **Energy Pills**: Locate all purple squares in the grey playable grid for energy refills (there are no energy refills in the first level)
- **Shape Changers**: Find all sets of 3 white squares for shape modification
- **Color Changers**: Find all 3x3 orange-containing grids for color modification
- **Door Location**: Note the black door square position

### 3. ROUTE OPTIMIZATION
- **Energy Budget**: Calculate energy needed for your planned route (LIMITED supply)
- **Changer Attempts**: Budget EXTRA energy for multiple changer attempts
- **Safety Margins**: Plan energy pill stops before running critically low
- **Efficient Pathing**: Minimize unnecessary movement between objectives

### 4. EXECUTION SEQUENCE
1. **Assess Energy**: Check if you have enough energy for the full sequence
2. **Shape First**: If shape is wrong, navigate to shape changer and cycle until correct
3. **Color Second**: If color is wrong, navigate to color changer and cycle until correct  
4. **Final Approach**: Once key matches door exactly, move to door for victory
5. **CRITICAL**: Move only within GREY playable squares - avoid wasting energy

## CHANGER CYCLING STRATEGY
- **Test and Evaluate**: Move onto changer, check if result is correct
- **Cycle if Wrong**: Move off changer, then back on to get next variation
- **Repeat as Needed**: Continue cycling until you get the required pattern
- **Energy Awareness**: Each cycle attempt costs energy - plan accordingly

## COMMON FAILURE MODES TO AVOID
1. **Energy Starvation**: Running out of energy before reaching objectives
2. **Premature Door Attempt**: Moving to door before key matches exactly
3. **Inefficient Cycling**: Not planning for multiple changer attempts
4. **Poor Route Planning**: Taking unnecessarily long paths between objectives

## CURRENT GAME STATE
- **Status**: {latest_frame.state.name}
- **Level**: {latest_frame.score} / 8
- **Grid Size**: {len(latest_frame.frame[0]) if latest_frame.frame and latest_frame.frame[0] else 0}x{len(latest_frame.frame[0][0]) if latest_frame.frame and latest_frame.frame[0] else 0}

## ANALYSIS CHECKLIST
Before each move:
1. **Energy**: How many purple squares remain at top? Safe to continue?
2. **Key Match**: Does current key (bottom-left) match door requirements?
3. **Next Objective**: Need energy, shape change, color change, or door approach?
4. **Route Safety**: Can I reach my objective and return to energy if needed?
5. **Backup Plan**: What if the changer doesn't give me the right result?

## CURRENT FRAME ANALYSIS
{self.pretty_print_3d(latest_frame.frame)}

# STRATEGIC DECISION
Analyze the situation using the framework above. Remember:
- Energy management is survival
- Key matching is victory condition  
- Efficient routing is optimization
- Multiple changer attempts are normal

What's your next move?
        """)

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(f"""
# GAME STATE UPDATE - STRATEGIC ANALYSIS

**Level**: {latest_frame.score} / 8
**Status**: {latest_frame.state.name}

## IMMEDIATE TACTICAL ASSESSMENT

### 1. ENERGY CRISIS CHECK
- **Purple Indicators**: How many purple squares at TOP of screen outside of the grey playable grid?
- **Danger Level**: Are you approaching energy starvation?
- **Energy Pills**: Where are purple squares in the grid for emergency refill?

### 2. KEY MATCHING ANALYSIS
- **Current Key**: Examine bottom-left corner - what shape and color?
- **Door Requirement**: Look at the black door - what's the target pattern inside?
- **Match Status**: 
  - Shape correct? Y or N
  - Color correct? Y or N
  - Ready for door? Y or N

### 3. OBJECTIVE PRIORITIZATION
Based on energy and key status:
- **URGENT**: Need energy refill immediately?
- **SHAPE**: Need to find and use shape changer (3 white squares)?
- **COLOR**: Need to find and use color changer (3x3 with orange)?
- **VICTORY**: Ready to approach the door?

### 4. TACTICAL PLANNING
- **Nearest Target**: What's the closest objective (energy/shape/color/door)?
- **Energy Cost**: Can you reach it safely?
- **Backup Route**: Where's the nearest energy pill if things go wrong?
- **Changer Strategy**: If using a changer, prepared for multiple attempts?

## VISUAL REFERENCE GUIDE
- **Your Position**: Look for the player icon in the grey playable grid
- **Energy Pills**: Purple squares scattered in the grey playable grid
- **Shape Changers**: 3 adjacent white squares
- **Color Changers**: 3x3 areas with orange elements
- **Door**: Black square with target pattern inside

## CURRENT GRID STATE
{self.pretty_print_3d(latest_frame.frame)}

## DECISION FRAMEWORK
1. **If Energy Low**: Head to nearest purple pill immediately
2. **If Shape Wrong**: Navigate to shape changer, cycle until correct
3. **If Color Wrong**: Navigate to color changer, cycle until correct
4. **If Key Matches**: Proceed directly to door for victory

**What's your tactical decision based on this analysis?**
        """)

    def build_functions(self) -> list[dict[str, Any]]:
        """Build function descriptions optimized for LockSmith tactical gameplay."""
        return [
            {
                "name": "RESET",
                "description": "Reset the current level. Use when energy is too low to continue or strategy has failed.",
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            },
            {
                "name": "ACTION1", 
                "description": "Move LEFT (west). Costs energy. Use to reach energy pills, shape/color changers, or door.",
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            },
            {
                "name": "ACTION2",
                "description": "Move RIGHT (east). Costs energy. Use to reach energy pills, shape/color changers, or door.", 
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            },
            {
                "name": "ACTION3",
                "description": "Move UP (north). Costs energy. Use to reach energy pills, shape/color changers, or door.",
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            },
            {
                "name": "ACTION4", 
                "description": "Move DOWN (south). Costs energy. Use to reach energy pills, shape/color changers, or door.",
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            },
            {
                "name": "ACTION5",
                "description": "Special action. Rarely needed in LockSmith. Costs energy - use cautiously.",
                "parameters": {
                    "type": "object", 
                    "properties": {}, 
                    "required": [], 
                    "additionalProperties": False
                }
            }
        ]

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format - converts functions to OpenAI tools format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools
