"""Central, secret-safe environment configuration for Trust Arena."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent


def _load() -> None:
    path = _ROOT / ".env"
    if not path.exists(): return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), value)
_load()


def _s(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, "").strip() or default


def _f(name: str, default: float) -> float:
    try: return float(_s(name, str(default)))
    except (TypeError, ValueError): return default


@dataclass(frozen=True)
class QuotaCeiling:
    per_minute: Optional[int] = None
    per_day: Optional[int] = None


@dataclass(frozen=True)
class Settings:
    server_url: Optional[str]
    team_id: Optional[str]
    team_token: Optional[str]
    primary_llm_provider: str
    fallback_llm_provider: str
    groq_api_key: Optional[str]
    gemini_api_key: Optional[str]
    groq_model: str
    gemini_model: str
    hard_deadline_seconds: float
    turn_budget_seconds: float
    submission_reserve_seconds: float
    groq_timeout_seconds: float
    gemini_timeout_seconds: float
    poll_interval_seconds: float
    total_rounds: int
    log_level: str
    log_format: str
    quota_ceilings: Dict[str, QuotaCeiling]

    def validate_startup(self) -> None:
        missing = []
        if not self.server_url: missing.append("SERVER_URL")
        if not self.team_id: missing.append("TEAM_ID")
        if not self.team_token: missing.append("TEAM_TOKEN")
        key = self.groq_api_key if self.primary_llm_provider == "groq" else self.gemini_api_key
        if not key: missing.append(self.primary_llm_provider.upper() + "_API_KEY")
        if self.primary_llm_provider == self.fallback_llm_provider:
            raise ValueError("PRIMARY_LLM_PROVIDER and FALLBACK_LLM_PROVIDER must differ")
        if missing: raise ValueError("Missing required environment configuration: " + ", ".join(missing))


_settings: Optional[Settings] = None
def get_settings(force_reload: bool = False) -> Settings:
    global _settings
    if _settings and not force_reload: return _settings
    primary = (_s("PRIMARY_LLM_PROVIDER", "groq") or "groq").lower()
    fallback = (_s("FALLBACK_LLM_PROVIDER", "gemini") or "gemini").lower()
    if primary not in ("groq", "gemini") or fallback not in ("groq", "gemini"):
        raise ValueError("Supported providers are groq and gemini")
    _settings = Settings(
        server_url=_s("SERVER_URL"), team_id=_s("TEAM_ID"), team_token=_s("TEAM_TOKEN"),
        primary_llm_provider=primary, fallback_llm_provider=fallback,
        groq_api_key=_s("GROQ_API_KEY"), gemini_api_key=_s("GEMINI_API_KEY") or _s("GOOGLE_API_KEY"),
        groq_model=_s("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b",
        gemini_model=_s("GEMINI_MODEL", "gemini-3.6-flash") or "gemini-3.6-flash",
        hard_deadline_seconds=_f("HARD_DEADLINE_SECONDS", 25), turn_budget_seconds=_f("TURN_BUDGET_SECONDS", 22),
        submission_reserve_seconds=_f("SUBMISSION_RESERVE_SECONDS", 5), groq_timeout_seconds=_f("GROQ_TIMEOUT_SECONDS", 8),
        gemini_timeout_seconds=_f("GEMINI_TIMEOUT_SECONDS", 5), poll_interval_seconds=_f("POLL_INTERVAL", 1),
        total_rounds=int(_f("TOTAL_ROUNDS", 7)), log_level=_s("LOG_LEVEL", "INFO") or "INFO",
        log_format=_s("LOG_FORMAT", "plain") or "plain",
        quota_ceilings={"groq": QuotaCeiling(), "gemini": QuotaCeiling()})
    return _settings
