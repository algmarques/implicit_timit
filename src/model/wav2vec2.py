"""
"""

from __future__ import annotations

from typing import Self, Any

from torch import Tensor
from torch import zeros

from torch.nn import Module, Parameter

from .projector import FeatureProjector
from .encoder import Encoder
from .quantizers import IsoGaussianVectorQuantizer, GaussianVectorQuantizer
from .quantizers import LinearVectorQuantizer

from ..utils import Arg
from .masking import get_masked_idxs, get_negative_idxs
from .losses import contrastive_loss
from .losses import log_likelihood, entropy, b_diversity, similarity


class Wav2Vec2(Module):
    """
    """

    _required = {
        "vq",
        "alpha",
        "beta",
        "gamma",
        "hidden_size",
        "codebook_dim",
        "n_groups",
        "masked_span_prob",
        "masked_span_length",
        "n_negatives",
        "detach",
        "freeze",
        "block",
        "gumbel_softmax"
    }

    def __init__(self: Wav2Vec2, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.feature_projector = FeatureProjector(**kwargs)

        if self.vq == "linear":
            self.vq = LinearVectorQuantizer(**kwargs)
        if self.vq == "iso_gaussian":
            self.vq = IsoGaussianVectorQuantizer(**kwargs)
        if self.vq == "gaussian":
            self.vq = GaussianVectorQuantizer(**kwargs)

        self.mask = Parameter(zeros(self.hidden_size), requires_grad=False)
        self.contrastive_encoder = Encoder(**kwargs)

    def contrastive_loss(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor,
        tau: float = 0.1
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

        negative_idxs = get_negative_idxs(masked_idxs, self.n_negatives)

        cloned_inputs = projected_inputs.clone()
        for i in range(batch_size):
            cloned_inputs[i, masked_idxs[i]] = self.mask

        if self.block:
            cloned_inputs = cloned_inputs.detach()

        hidden_states = self.contrastive_encoder(cloned_inputs, attention_mask)

        c_loss = contrastive_loss(
            hidden_states,
            codevectors,
            masked_idxs,
            negative_idxs,
            tau
        )

        return c_loss

    def log_likelihood(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        projected_inputs = self.feature_projector(inputs, attention_mask)
        if self.detach:
            projected_inputs = projected_inputs.detach()

        return log_likelihood(projected_inputs, attention_mask, self.vq)

    def entropy(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        projected_inputs = self.feature_projector(inputs, attention_mask)
        if self.detach:
            projected_inputs = projected_inputs.detach()

        return entropy(projected_inputs, attention_mask, self.vq)

    def b_diversity(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        projected_inputs = self.feature_projector(inputs, attention_mask)
        if self.detach:
            projected_inputs = projected_inputs.detach()

        return b_diversity(projected_inputs, attention_mask, self.vq)

    def similarity(self: Wav2Vec2) -> Tensor:
        """
        """

        return similarity(self.vq)

    def loss(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor,
        target: Tensor | None,
    ) -> Tensor:
        """
        """

        c_loss = self.contrastive_loss(inputs, attention_mask)

        ll_loss = 0.0
        if self.alpha:
            ll_loss = self.log_likelihood(inputs, attention_mask) * self.alpha

        e_loss = 0.0
        if self.beta:
            e_loss = self.entropy(inputs, attention_mask) * self.beta

        s_loss = 0.0
        if self.gamma:
            s_loss = self.similarity() * self.gamma

        return c_loss + ll_loss + e_loss + s_loss


    def forward(
        self: Wav2Vec2,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        projected_inputs = self.feature_projector(inputs, attention_mask)
        hidden_states = self.contrastive_encoder(
            projected_inputs,
            attention_mask
        )

        return hidden_states


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
