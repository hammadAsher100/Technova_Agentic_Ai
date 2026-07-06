"""Common types and interface every provider client must implement.

ModelRouter depends only on this interface (BaseProviderClient.complete)
and these three dataclasses — never on a specific provider's SDK or
REST response shape. That is what lets the Section 5 fallback chain
swap providers with zero caller-side changes, and it is also what
makes swapping any single provider's transport implementation (e.g.
raw HTTP -> official SDK, see docs/ARCHITECTURE.md) a contained,
single-file change.

Python 3.9 note (REG-02): every annotation below uses typing.Optional /
typing.List / typing.Dict rather than the 3.10+ `X | Y` syntax.
"""
import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ComplexityTier(Enum):
    """Section 5: 'reasoning complexity required' is one of the signals
    ModelRouter uses to pick a provider. LOW/MEDIUM route to the fast,
    cheap workhorse providers; HIGH is reserved for the provider(s)
    explicitly flagged for it (e.g. Gemini Pro) to protect their tighter
    quota."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.3
    complexity: ComplexityTier = ComplexityTier.MEDIUM
    # Free-form hints for the router (e.g. estimated input token count).
    # Not consumed by the generic routing logic yet — a documented hook
    # for Phase 1 refinement rather than unused speculative plumbing.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class BaseProviderClient(abc.ABC):
    """Every concrete provider client (GroqClient, GeminiClient, ...)
    subclasses this and implements `complete`. `provider_id` may be set
    as a class attribute (Groq/Mistral/OpenRouter — fixed per class) or
    an instance attribute (Gemini — one class, two profiles: flash/pro)."""

    provider_id: str

    @abc.abstractmethod
    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        """Execute one completion call, bounded by `timeout` seconds.

        Must raise a subclass of agent.exceptions.ProviderError on any
        failure — never let a raw urllib/socket exception escape.
        """
        raise NotImplementedError
