"""
"""

from typing import Never

from torch import Tensor


def is_infinitesimal(eps: float) -> None:
    """
    """

    if eps < 0.0 or eps > 1.0:
        raise ValueError()


def attention_mask_validation(
    hidden_state: Tensor,
    attention_mask: Tensor | None
) -> Never:
    """
    """

    if attention_mask is not None:
        if not len(hidden_state.shape) > len(attention_mask.shape):
            raise ValueError()

        for hs_axis, am_axis in zip(hidden_state.shape, attention_mask.shape):
            if hs_axis != am_axis:
                raise ValueError()


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
