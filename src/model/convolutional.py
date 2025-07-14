"""
"""

from __future__ import annotations

from functools import partial

from torch import Tensor
from torch.nn.functional import pad
from torch.nn import Module, Conv1d, Dropout, GLU, SiLU

from ..utils import Arg
from .layer_norm import MaskedLayerNorm

_UnbiasedConv = partial(Conv1d, bias=False)
_PointwiseConv = partial(_UnbiasedConv, kernel_size=1)


class Convolutional(Module):
    """
    """

    _required = {
        "hidden_size",
        "conv_kernel_size",
        "conv_dropout",
        "layer_norm_eps"
    }

    def __init__(self: Convolutional, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        if (self.conv_kernel_size - 1) % 2 == 1:
            raise ValueError()

        self.layer_norm = MaskedLayerNorm(self.layer_norm_eps)

        self.pre_pointwise = _PointwiseConv(
            self.hidden_size,
            2 * self.hidden_size,
        )

        self.glu = GLU(dim=1)

        self.depthwise = _UnbiasedConv(
            self.hidden_size,
            self.hidden_size,
            self.conv_kernel_size,
            groups=self.hidden_size,
        )

        self.depthwise_layer_norm = MaskedLayerNorm(self.layer_norm_eps)

        self.silu = SiLU()

        self.post_pointwise = _PointwiseConv(
            self.hidden_size,
            self.hidden_size,
        )

        self.dropout = Dropout(self.conv_dropout)

    def forward(
        self: Convolutional,
        hidden_state: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        hidden_state = self.layer_norm(hidden_state, attention_mask)
        hidden_state = hidden_state * attention_mask[..., None]

        hidden_state = hidden_state.transpose(1, 2)

        hidden_state = self.pre_pointwise(hidden_state)
        hidden_state = self.glu(hidden_state)

        hidden_state = pad(hidden_state, (self.conv_kernel_size - 1, 0))

        hidden_state = self.depthwise(hidden_state)

        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = self.depthwise_layer_norm(hidden_state, attention_mask)
        hidden_state = hidden_state.transpose(1, 2)

        hidden_state = self.silu(hidden_state)
        hidden_state = self.post_pointwise(hidden_state)

        hidden_state = self.dropout(hidden_state)

        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = hidden_state * attention_mask[..., None]

        return hidden_state


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
