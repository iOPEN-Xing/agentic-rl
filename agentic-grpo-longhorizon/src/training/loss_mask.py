"""Loss-mask policy selection kept independent from tokenizer dependencies."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LOSS_MASK_MODES = {"all_assistant", "last_assistant"}


def select_assistant_indices(
    messages: Sequence[Mapping[str, Any]], loss_mask_mode: str
) -> list[int]:
    """Return assistant turns to supervise under the requested masking policy."""

    if loss_mask_mode not in LOSS_MASK_MODES:
        raise ValueError(
            f"unknown loss_mask_mode={loss_mask_mode!r}; "
            f"valid modes: {sorted(LOSS_MASK_MODES)}"
        )
    indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if loss_mask_mode == "last_assistant" and indices:
        return [indices[-1]]
    return indices
