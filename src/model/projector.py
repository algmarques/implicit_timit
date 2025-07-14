"""
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import Module, Linear

from ..utils import Arg
from .layer_norm import MaskedLayerNorm


class FeatureProjector(Module):
    """
    """

    _required = {
        "hidden_size",
        "feat_proj_input_dim",
        "layer_norm_eps"
    }

    def __init__(self: FeatureProjector, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.pre_layer_norm = MaskedLayerNorm(self.layer_norm_eps)
        self.linear = Linear(self.feat_proj_input_dim, self.hidden_size)
        self.post_layer_norm = MaskedLayerNorm(self.layer_norm_eps)

    def forward(
        self: FeatureProjector,
        hidden_state: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        hidden_state = self.pre_layer_norm(hidden_state, attention_mask)
        hidden_state = self.linear(hidden_state)
        hidden_state = self.post_layer_norm(hidden_state, attention_mask)

        return hidden_state


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
