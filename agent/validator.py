"""Output validation scaffold (Section 0, step 4).

Section 0 describes this file's structure as generic even though
task-specific validation rules may be layered in during Phase 1. The
`register()` / `validate()` pattern below is what makes that true:
Phase 1 adds task-specific checks by registering additional validator
functions against a running `Validator` instance, without touching the
core validation flow defined here.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agent.providers.base import LLMResponse
from agent.safety import contains_any

logger = logging.getLogger(__name__)

# Returns a list of error strings; an empty list means "no problems found".
ValidatorFn = Callable[[LLMResponse], List[str]]


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)


class Validator:
    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 8000,
        forbidden_snippets: Optional[List[str]] = None,
    ) -> None:
        self._min_length = min_length
        self._max_length = max_length
        self._forbidden_snippets = forbidden_snippets or []
        self._extra_validators: List[ValidatorFn] = []

    def register(self, validator_fn: ValidatorFn) -> None:
        """Add a task-specific validator (Phase 1). Runs after the
        generic checks below, in registration order."""
        self._extra_validators.append(validator_fn)

    def validate(self, response: LLMResponse) -> ValidationResult:
        errors: List[str] = []
        text = response.text or ""

        if len(text.strip()) < self._min_length:
            errors.append(f"response text shorter than minimum length {self._min_length}")
        if len(text) > self._max_length:
            errors.append(f"response text longer than maximum length {self._max_length}")

        leaked = contains_any(text, self._forbidden_snippets)
        if leaked:
            errors.append(f"response contains forbidden internal snippet(s): {leaked}")

        for validator_fn in self._extra_validators:
            errors.extend(validator_fn(response))

        is_valid = not errors
        if not is_valid:
            logger.warning("validator.rejected", extra={"errors": errors, "provider": response.provider})
        return ValidationResult(is_valid=is_valid, errors=errors)
