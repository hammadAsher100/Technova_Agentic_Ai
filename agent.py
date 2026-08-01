"""TECHNOVA 2026 Trust Arena official executable entry point."""
import logging
import time
from typing import Dict, Tuple

import requests

from agent.core import decide as strategy_decide
from config.logging_config import setup_logging
from config.settings import get_settings

logger = logging.getLogger("trust_arena")


def decide(game_state: dict) -> Tuple[str, str, str]:
    """Official interface: return (decision, message, concise reasoning)."""
    return strategy_decide(game_state)


def _urls(server_url: str, practice: bool) -> Tuple[str, str]:
    prefix = "/practice" if practice else ""
    base = server_url.rstrip("/")
    return base + prefix + "/my-turn", base + prefix + "/my-move"


def _submit(session: requests.Session, url: str, params: Dict[str, str], headers: Dict[str, str],
            payload: Tuple[str, str, str], deadline: float) -> requests.Response:
    remaining = max(0.2, deadline - time.monotonic())
    body = {"decision": payload[0], "message": payload[1], "reasoning": payload[2]}
    return session.post(url, params=params, headers=headers, json=body, timeout=min(5.0, remaining))


def main() -> None:
    settings = get_settings()
    settings.validate_startup()
    setup_logging(settings.log_level, settings.log_format)
    logger.info("Agent starting for team %s", settings.team_id)  # never log token or keys
    logger.info("Polling every %.1fs", settings.poll_interval_seconds)
    practice_mode = False
    session = requests.Session()
    headers = {"X-Team-Token": settings.team_token or ""}
    params = {"team_id": settings.team_id or ""}
    while True:
        try:
            turn_url, move_url = _urls(settings.server_url or "", practice_mode)
            response = session.get(turn_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            state = response.json()
            status = state.get("status")
            practice_mode = bool(state.get("practice_mode", False))
            if practice_mode:
                turn_url, move_url = _urls(settings.server_url or "", True)
            if status == "wait":
                time.sleep(float(state.get("retry_in") or settings.poll_interval_seconds)); continue
            if status == "your_turn":
                received = time.monotonic()
                deadline = received + settings.turn_budget_seconds
                if state.get("test_mode"):
                    payload = ("cooperate", "", "Test mode mandated cooperation.")
                else:
                    payload = strategy_decide(state, deadline)
                move_response = _submit(session, move_url, params, headers, payload, deadline)
                move_response.raise_for_status()
                accepted = move_response.json().get("accepted")
                logger.info("Move submission accepted=%s round=%s", bool(accepted), state.get("round_num"))
            elif status == "match_complete":
                logger.info("Match complete; waiting for assignment")
            time.sleep(settings.poll_interval_seconds)
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot reach server; retrying in 3s"); time.sleep(3)
        except requests.exceptions.Timeout:
            logger.warning("Request timed out; cached move will be reused if turn repeats")
            time.sleep(settings.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Agent stopped"); return
        except Exception as exc:
            logger.error("Runtime error (%s); retrying in 3s", type(exc).__name__)
            time.sleep(3)


if __name__ == "__main__":
    main()
