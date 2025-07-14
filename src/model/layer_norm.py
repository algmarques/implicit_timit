"""
"""

from __future__ import annotations

from typing import Never

from torch import Tensor, Size
from torch import ones, zeros
from torch.nn import Module, Parameter

from .validation import is_infinitesimal, attention_mask_validation


def masked_layer_norm(
    hidden_state: Tensor,
    attention_mask: Tensor | None,
    eps: float = 1e-5
) -> Tensor:
    """
    """

    is_infinitesimal(eps)
    attention_mask_validation(hidden_state, attention_mask)

    cloned_hidden_state = hidden_state.clone()
    tmp = hidden_state[attention_mask]
    mean = tmp.mean()
    eps_std = (tmp.var() + eps).sqrt()

    cloned_hidden_state[attention_mask] = (tmp - mean) / eps_std

    return cloned_hidden_state


class MaskedLayerNorm(Module):
    """
    """

    def __init__(self: MaskedLayerNorm, eps: float = 1e-5) -> None:
        """
        """

        super().__init__()
        self.eps = eps

    def forward(
        self: MaskedLayerNorm,
        hidden_state: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        hidden_state = masked_layer_norm(
            hidden_state,
            attention_mask,
            self.eps
        )

        return hidden_state


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
