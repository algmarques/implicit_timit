"""
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import Module, Linear, Dropout, SiLU

from ..utils import Arg


class FeedForward(Module):
    """
    """

    _required = {
        "hidden_size",
        "hidden_dropout",
        "intermediate_size",
        "feed_forward_dropout"
    }

    def __init__(self: FeedForward, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.pre_linear = Linear(self.hidden_size, self.intermediate_size)
        self.silu = SiLU()
        self.pre_dropout = Dropout(self.feed_forward_dropout)

        self.post_linear= Linear(self.intermediate_size, self.hidden_size)
        self.post_dropout = Dropout(self.hidden_dropout)

    def forward(
        self: FeedForward,
        hidden_state: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        hidden_state = hidden_state * attention_mask[..., None]

        cloned_hidden_state = hidden_state.clone()
        tmp_state = hidden_state[attention_mask]

        tmp_state = self.pre_linear(tmp_state)
        tmp_state = self.silu(tmp_state)
        tmp_state = self.pre_dropout(tmp_state)

        tmp_state = self.post_linear(tmp_state)
        tmp_state = self.post_dropout(tmp_state)

        cloned_hidden_state[attention_mask] = tmp_state

        return cloned_hidden_state


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
