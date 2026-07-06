"""Central exception hierarchy for the agent.

Every module raises one of these instead of letting a raw stdlib
exception (urllib.error.*, KeyError, etc.) leak across module
boundaries. This is what lets agent/model_router.py distinguish
retryable transport failures from non-retryable ones (e.g. an auth
error should never be retried against the same provider, but a
timeout should count against that provider's health score) without
knowing anything about urllib.
"""


class AgentError(Exception):
    """Base class for every exception this codebase raises on purpose."""


class ProviderError(AgentError):
    """Base class for provider-transport-level failures."""


class ProviderAuthError(ProviderError):
    """API key missing, invalid, or rejected (HTTP 401 / 403)."""


class ProviderRateLimitError(ProviderError):
    """Provider returned HTTP 429 (quota or rate limit exceeded)."""


class ProviderTimeoutError(ProviderError):
    """Request did not complete within the allotted per-call timeout."""


class ProviderUnavailableError(ProviderError):
    """Connection-level failure: DNS, refused connection, network unreachable."""


class ProviderInvalidResponseError(ProviderError):
    """Provider responded, but the body didn't match the expected shape."""


class AllProvidersExhaustedError(AgentError):
    """Every candidate provider failed, or the soft deadline was reached.

    Raised by ModelRouter.complete(). Section 5 of the spec makes this
    the planner's problem, not the router's: catching this and falling
    back to a local heuristic / neutral action is a Phase 1 concern
    (agent/planner.py, agent/core.py), not something model_router.py
    should decide on its own.
    """


class ValidationError(AgentError):
    """Raised when an LLM response fails agent/validator.py's checks."""
