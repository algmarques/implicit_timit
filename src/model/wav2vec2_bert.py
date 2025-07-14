"""
"""

from __future__ import annotations

from torch import Tensor

from ..utils import Arg
from .encoder import Encoder
from .wav2vec2 import Wav2Vec2
from .losses import mlm_loss
from .masking import get_masked_idxs


class Wav2Vec2Bert(Wav2Vec2):
    """
    """

    _required = Wav2Vec2._required | {"zeta"}

    def __init__(self: Wav2Vec2Bert, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)

        self.mlm_encoder = Encoder(**kwargs)

    def mlm_loss(
        self: Wav2Vec2Bert,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        batch_size = len(inputs)
        projected_inputs = self.feature_projector(inputs, attention_mask)
        codevectors, codewords = self.vq(projected_inputs, attention_mask)

        masked_idxs = get_masked_idxs(
            projected_inputs,
            self.masked_span_prob,
            self.masked_span_length,
            attention_mask
        )

        cloned_inputs = projected_inputs.clone()
        for i in range(batch_size):
            cloned_inputs[i, masked_idxs[i]] = self.mask

        if self.block:
            cloned_inputs = cloned_inputs.detach()

        hidden_states = self.contrastive_encoder(cloned_inputs, attention_mask)
        hidden_states = self.mlm_encoder(hidden_states, attention_mask)

        return mlm_loss(hidden_states, codewords, masked_idxs)

    def loss(
        self: Wav2Vec2Bert,
        inputs: Tensor,
        attention_mask: Tensor,
        target: Tensor | None,
    ) -> Tensor:
        """
        """

        loss = super().loss(inputs, attention_mask, target)

        mlm_loss = 0.0
        if self.zeta:
            mlm_loss = self.mlm_loss(inputs, attention_mask) * self.zeta

        return loss + mlm_loss

    def forward(
        self: Wav2Vec2Bert,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        hidden_states = super().forward(inputs, attention_mask)
        hidden_states = self.mlm_encoder(hidden_states, attention_mask)

        return hidden_states


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
