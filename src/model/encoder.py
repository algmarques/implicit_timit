"""
"""

from __future__ import annotations

from random import random

from torch import Tensor
from torch.nn import Module, ModuleList, Dropout

from ..utils import Arg
from .layer_norm import MaskedLayerNorm
from .feed_forward import FeedForward
from .convolutional import Convolutional
from .self_attention import SelfAttention


class Layer(Module):
    """
    """

    _required = {
        "hidden_size",
        "attention_dropout",
        "layer_norm_eps"
    }

    def __init__(self: Layer, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.pre_ff_layer_norm = MaskedLayerNorm(self.layer_norm_eps)
        self.pre_ff = FeedForward(**kwargs)

        self.attention_layer_norm = MaskedLayerNorm(self.layer_norm_eps)
        self.attention_dropout = Dropout(self.attention_dropout)
        self.attention = SelfAttention(**kwargs)

        self.convolutional = Convolutional(**kwargs)

        self.post_ff_layer_norm = MaskedLayerNorm(self.layer_norm_eps)
        self.post_ff = FeedForward(**kwargs)

        self.layer_norm = MaskedLayerNorm(self.layer_norm_eps)

    def forward(
        self: Layer,
        hidden_state: Tensor,
        attention_mask: Tensor,
        conv_attention_mask: Tensor,
    ) -> None:
        """
        """

        residual = hidden_state
        hidden_state = self.pre_ff_layer_norm(hidden_state, attention_mask)
        hidden_state = self.pre_ff(hidden_state, attention_mask)
        hidden_state = residual + hidden_state / 2

        residual = hidden_state
        hidden_state = self.attention_layer_norm(hidden_state, attention_mask)
        hidden_state, attention_weigts = self.attention(
            hidden_state,
            attention_mask
        )
        hidden_state = self.attention_dropout(hidden_state)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.convolutional(hidden_state, conv_attention_mask)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.post_ff_layer_norm(hidden_state, attention_mask)
        hidden_state = self.post_ff(hidden_state, attention_mask)
        hidden_state = residual + hidden_state / 2

        hidden_state = self.layer_norm(hidden_state, attention_mask)

        return hidden_state, attention_weigts


class Encoder(Module):
    """
    """

    _required = {
        "position_embeddings_type",
        "hidden_dropout",
        "n_hidden_layers",
        "layer_dropout"
    }

    def __init__(self: Encoder, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.dropout = Dropout(self.hidden_dropout)
        self.layers = ModuleList(
            [Layer(**kwargs) for _ in range(self.n_hidden_layers)]
        )

    def forward(
        self: Encoder,
        hidden_state: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        hidden_state = hidden_state * attention_mask[..., None]
        hidden_state = self.dropout(hidden_state)

        for layer in self.layers:
            if not self.training or random() > self.layer_dropout:
                hidden_state, _ = layer(
                    hidden_state,
                    attention_mask,
                    attention_mask,
                )

        return hidden_state


def main() -> None:
    """
    """


if __name__ == "__main__":
    main()
