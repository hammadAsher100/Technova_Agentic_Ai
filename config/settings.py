"""Centralized, environment-driven configuration.

Every rule-governed or environment-specific parameter (REG-10, REG-03)
lives here — nowhere else in the codebase should read os.environ
directly or hardcode a timeout / quota / model-name constant.

Loads `.env` via a small hand-rolled parser rather than `python-dotenv`.
This is a deliberate consequence of the Phase 0 dependency decision
documented in docs/ARCHITECTURE.md ("Dependency strategy"): the agent's
runtime has zero third-party dependencies so a Python-3.9 `pip install`
can never fail on the judges' machine. `.env` parsing is ~15 lines of
stdlib code, so hand-rolling it here was a clear win regardless of
whether python-dotenv itself would have been 3.9-safe.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _load_dotenv(path: Path) -> None:
    """KEY=VALUE per line, '#' comments, optional quoting. Existing
    environment variables always win over the file, matching
    python-dotenv's default (non-override) behavior."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(_ENV_FILE)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("settings.invalid_float_env", extra={"name": name, "value": raw})
        return default


def _env_int_optional(name: str) -> Optional[int]:
    """Unlike _env_float above, this has no numeric default — quota
    ceilings are deliberately None (not locally enforced) unless the
    team fills them in from their own provider dashboards. See
    .env.example: free-tier limits vary by model and change over time,
    so no specific number is assumed here."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("settings.invalid_int_env", extra={"name": name, "value": raw})
        return None


@dataclass(frozen=True)
class QuotaCeiling:
    per_minute: Optional[int] = None
    per_day: Optional[int] = None


@dataclass(frozen=True)
class Settings:
    # --- Competition-mandated timing (REG-06: 25s hard limit/round) ---
    hard_deadline_seconds: float
    soft_deadline_seconds: float
    per_call_cap_seconds: float

    # --- API keys (never logged raw — see config/logging_config.py redact()) ---
    groq_api_key: Optional[str]
    gemini_api_key: Optional[str]
    mistral_api_key: Optional[str]
    openrouter_api_key: Optional[str]

    # --- Model selection (Section 5 defaults; override via .env) ---
    groq_model: str
    gemini_flash_model: str
    gemini_pro_model: str
    mistral_model: str
    openrouter_model: Optional[str]
    openrouter_referer: Optional[str]
    openrouter_title: Optional[str]

    # --- Free-tier quota ceilings (Section 8); None = not locally enforced ---
    quota_ceilings: Dict[str, QuotaCeiling]

    # --- Logging ---
    log_level: str
    log_format: str  # "plain" | "json"

    def configured_providers(self) -> List[str]:
        """Provider keys that have a non-empty API key configured."""
        mapping = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "mistral": self.mistral_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return [name for name, key in mapping.items() if key]


_settings_singleton: Optional[Settings] = None


def get_settings(force_reload: bool = False) -> Settings:
    """Process-wide Settings singleton.

    `force_reload=True` exists for tests that mutate os.environ and
    need to re-read it; production code should never need it.
    """
    global _settings_singleton
    if _settings_singleton is not None and not force_reload:
        return _settings_singleton

    _settings_singleton = Settings(
        hard_deadline_seconds=_env_float("HARD_DEADLINE_SECONDS", 25.0),
        soft_deadline_seconds=_env_float("SOFT_DEADLINE_SECONDS", 13.0),
        per_call_cap_seconds=_env_float("PER_CALL_CAP_SECONDS", 8.0),
        groq_api_key=_env_str("GROQ_API_KEY"),
        gemini_api_key=_env_str("GEMINI_API_KEY") or _env_str("GOOGLE_API_KEY"),
        mistral_api_key=_env_str("MISTRAL_API_KEY"),
        openrouter_api_key=_env_str("OPENROUTER_API_KEY"),
        groq_model=_env_str("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_flash_model=_env_str("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
        gemini_pro_model=_env_str("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
        mistral_model=_env_str("MISTRAL_MODEL", "mistral-large-latest"),
        openrouter_model=_env_str("OPENROUTER_MODEL"),
        openrouter_referer=_env_str("OPENROUTER_SITE_URL"),
        openrouter_title=_env_str("OPENROUTER_SITE_NAME"),
        quota_ceilings={
            "groq": QuotaCeiling(
                per_minute=_env_int_optional("GROQ_RPM_CEILING"),
                per_day=_env_int_optional("GROQ_RPD_CEILING"),
            ),
            "gemini": QuotaCeiling(
                per_minute=_env_int_optional("GEMINI_RPM_CEILING"),
                per_day=_env_int_optional("GEMINI_RPD_CEILING"),
            ),
            "mistral": QuotaCeiling(
                per_minute=_env_int_optional("MISTRAL_RPM_CEILING"),
                per_day=_env_int_optional("MISTRAL_RPD_CEILING"),
            ),
            "openrouter": QuotaCeiling(
                per_minute=_env_int_optional("OPENROUTER_RPM_CEILING"),
                per_day=_env_int_optional("OPENROUTER_RPD_CEILING"),
            ),
        },
        log_level=_env_str("LOG_LEVEL", "INFO"),
        log_format=_env_str("LOG_FORMAT", "plain"),
    )
    return _settings_singleton
