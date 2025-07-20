# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **LockSmithExpert Agent Integration** (2025-01-18)
  - Integrated new LockSmithExpert agent specialized for the LockSmith game
  - Added import and registration in agents/__init__.py for the locksmithexpert agent
  - Agent features advanced energy management, key modification strategies, and tactical gameplay analysis
  - **Technical Details**: The agent extends the LLM base class with specialized prompts for LockSmith game mechanics including energy conservation, shape/color changer cycling, and strategic route planning
  - **Modified Files**: `agents/__init__.py` (import and registration)
  - **Available Agent**: Can now be used with `--agent=locksmithexpert`

### Fixed
- **LockSmithExpert Recording Issue Fix** (2025-01-18)
  - Fixed issue where LockSmithExpert agent runs weren't being recorded in the scorecard
  - Added proper initialization of reasoning-related instance variables in LockSmithExpert.__init__()
  - Added explicit choose_action() and track_tokens() method overrides to ensure reasoning metadata capture
  - Enhanced reasoning metadata with LockSmith-specific context (agent_type, game_specialization)
  - **Technical Details**: The agent was inheriting from ReasoningLLM but wasn't properly initializing reasoning capture variables (_last_reasoning_tokens, _last_response_content, _total_reasoning_tokens). Without explicit initialization and method overrides, the recording system couldn't capture the reasoning metadata properly. Added explicit __init__, choose_action, and track_tokens methods following the same pattern as GuidedLLM.
  - **Modified Files**: `agents/templates/locksmith_agent.py` (added missing recording infrastructure)
  - **Root Cause**: Missing reasoning capture variable initialization and method overrides needed for proper recording system integration

### Fixed
- **Agent Attempt Recording Fix** (2025-01-18)
  - Fixed issue where agent attempts weren't being recorded when games didn't complete successfully
  - Changed cleanup logic to always record agent's own attempt data regardless of API scorecard status
  - Eliminated "No data found for game" warnings for incomplete attempts (which are expected)
  - **Technical Details**: Previously, recording only happened when the API's final scorecard contained complete game data. For incomplete games (hitting MAX_ACTIONS, timeouts, etc.), the API wouldn't include that game in the scorecard, so no summary was recorded. Now agents record their own attempt summary with score, actions taken, duration, exit reason, etc. regardless of API scorecard status.
  - **Modified Files**: `agents/agent.py` (cleanup method)
  - **Root Cause**: Dependency on API scorecard data that's only available for successfully completed games
- **LockSmithExpert reasoning capture**: Changed LockSmithExpert inheritance from `LLM` to `ReasoningLLM` and added `REASONING_EFFORT = "high"` to enable reasoning metadata capture like the ReasoningAgent. This allows the locksmith agent to record reasoning tokens and thought processes for analysis. (Claude Sonnet 4)
- **LockSmithExpert Game Mechanics Fix** (2025-01-18)
  - Fixed agent misunderstanding of energy system and playable area boundaries
  - Corrected prompt to understand purple squares are UI indicators, not collectible energy pills
  - Clarified that only grey squares are moveable within the playable area
  - Updated strategy to work with finite energy instead of trying to collect non-existent refills
  - **Technical Details**: Agent was trying to "collect" purple energy indicator squares from the UI, thinking they were in-game energy pills. Fixed by clarifying energy system has no refills and playable area is limited to grey squares only.
  - **Modified Files**: `agents/templates/locksmith_agent.py` (energy system, resource mapping, execution sequence)
  - **Root Cause**: Prompt confusion between game UI elements and interactive game objects
- **Scorecard Race Condition Fix** (2025-01-18)
  - Fixed "card_id not found" errors by eliminating duplicate cleanup calls that caused race condition
  - Removed agent's individual cleanup call to prevent concurrent scorecard access with swarm
  - Added resilient error handling and enhanced debugging for scorecard request failures
  - **Technical Details**: Agents were calling cleanup() twice - once individually from agent.main() and once from swarm.cleanup(). The individual cleanup tried to fetch scorecard from API while swarm was simultaneously closing it. Fixed by removing duplicate cleanup call and letting swarm handle cleanup with proper scorecard.
  - **Modified Files**: `agents/agent.py` (removed duplicate cleanup call, enhanced error handling)
  - **Root Cause**: Duplicate cleanup execution causing race condition between agent and swarm scorecard operations
- **Windows Encoding Fix for ARC-AGI-3 API** (2025-01-18)
  - Fixed "SERVER_ERROR" and "card_id not found" errors occurring only on Windows systems
  - Removed unnecessary JSON serialization roundtrips causing encoding mismatches between Windows (cp1252) and API (UTF-8)
  - Added explicit UTF-8 Content-Type headers for HTTP requests as additional safety measure
  - **Technical Details**: Windows Python defaults to cp1252 encoding while ARC-AGI-3 API expects UTF-8. The `json.dumps(data)` → `json.loads(json_str)` roundtrip caused encoding corruption when LLM responses contained Unicode characters (em-dashes, smart quotes). Fixed by passing Python dictionaries directly to requests' `json=` parameter, ensuring UTF-8 encoding.
  - **Modified Files**: `agents/agent.py` (do_action_request), `agents/swarm.py` (scorecard methods)
  - **Root Cause**: Platform-specific character encoding differences between Windows cp1252 and Unix UTF-8 defaults
- **LockSmithExpert Context Length Fix** (2025-01-18)
  - Fixed context length exceeded error by switching from gpt-4o-mini-2024-07-18 to o4-mini-2025-04-16
  - Updated model to support larger context window for detailed LockSmith strategic prompts
  - Added missing imports (json, logging, os, openai) to match other LLM agents
  - **Technical Details**: The detailed tactical prompts (135K tokens) exceeded gpt-4o-mini's 128K limit. o4-mini-2025-04-16 provides larger context capacity for comprehensive game analysis
  - **Modified Files**: `agents/templates/locksmith_agent.py` (model change and imports)
  - **Resolved Error**: `context_length_exceeded: 135175 tokens vs 128000 limit`
- **LockSmithExpert Tool Compatibility Fix** (2025-01-18)
  - Fixed OpenAI BadRequestError in LockSmithExpert agent due to missing build_tools() method
  - Added proper build_tools() method to convert custom functions to OpenAI tools format
  - **Technical Details**: Agent had MODEL_REQUIRES_TOOLS=True but only overrode build_functions(), causing base class to use default tools instead of custom LockSmith-specific function descriptions
  - **Modified Files**: `agents/templates/locksmith_agent.py` (added build_tools method)
  - **Resolved Error**: `BadRequestError: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'`
- **LockSmithExpert IndexError Fix** (2025-01-18)
  - Fixed IndexError in LockSmithExpert agent when accessing empty frame data
  - Added proper bounds checking for grid size calculation in build_user_prompt method
  - **Technical Details**: The agent was trying to access `latest_frame.frame[0]` without checking if frame data exists first, causing crashes when frame is empty or malformed
  - **Modified Files**: `agents/templates/locksmith_agent.py` (grid size calculation)
  - **Resolved Error**: `IndexError: list index out of range` on frame[0] access
- **Windows Unicode Encoding Issue** (2025-01-18)
  - Fixed UnicodeEncodeError when logging AI assistant responses containing Unicode characters (like en-dash ‐)
  - Added UTF-8 encoding configuration for logging handlers on Windows systems
  - Improved cross-platform compatibility with fallback encoding handling
  - **Technical Details**: The issue occurred when AI responses contained Unicode characters that Windows' default cp1252 codec couldn't handle. The fix configures both stdout and file logging handlers to use UTF-8 encoding with error replacement.
  - **Modified Files**: `main.py` (logging configuration)
  - **Resolved Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2010'`

---
*Changes documented by Claude Sonnet 4* 