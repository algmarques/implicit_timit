"""
"""

from __future__ import annotations

from torch import Tensor
from torch import int64, float32
from torch import zeros, zeros_like, ones, rand
from torch import cat
from torch.nn import Linear, LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import one_hot, softmax

from ..utils import Arg
from .wav2vec2 import Wav2Vec2
from .losses import e2e_loss


class Wav2Vec2E2E(Wav2Vec2):
    """
    """

    _required = Wav2Vec2._required
    _required |= {
        "sigma",
        "teacher_forcing",
        "n_classes",
        "n_layers",
        "decoding_range"
    }

    def __init__(self: Wav2Vec2BertE2E, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)

        self.omega = 1.0

        self.lstm_encoder = LSTM(self.hidden_size, self.hidden_size,
            num_layers=self.n_layers, bias=False)
        self.lstm_decoder = LSTM(self.hidden_size, self.hidden_size,
            num_layers=self.n_layers, bias=False)

        self.pre_linear = Linear(self.n_classes, self.hidden_size)
        self.post_linear = Linear(self.hidden_size, self.n_classes)

    def e2e_loss(
        self: Wav2Vec2BertE2E,
        inputs: Tensor,
        attention_mask: Tensor,
        target: Tensor | None,
    ) -> Tensor:
        """
        """

        batch_size = len(inputs)
        proj_inputs = self.feature_projector(inputs, attention_mask)

        lengths = attention_mask.sum(dim=-1).to("cpu")

        proj_inputs = proj_inputs.transpose(0, 1)
        packed_inputs = pack_padded_sequence(
            proj_inputs, lengths, enforce_sorted=False)

        _, context = self.lstm_encoder(packed_inputs)

        target = one_hot(target, num_classes=self.n_classes).to(float32)
        target = target.transpose(0, 1)
        max_seq_length = len(target)

        logits = zeros_like(target)
        logits[0] = target[0]

        for i in range(max_seq_length - 1):

            tmp = zeros_like(target)
            mask = rand(max_seq_length, batch_size) < self.omega
            tmp[mask] = target[mask]
            tmp[~mask] = logits[~mask]
            tmp = tmp[0: i + 1]

            tmp = self.pre_linear(tmp)
            tmp, _ = self.lstm_decoder(tmp, context)
            tmp = self.post_linear(tmp)
            tmp = softmax(tmp, dim=-1)
            logits[i + 1] = tmp[i]

        logits = logits.reshape(-1, self.n_classes)
        target = target.reshape(-1, self.n_classes)

        return e2e_loss(logits, target)

    def loss(
        self: Wav2Vec2BertE2E,
        inputs: Tensor,
        attention_mask: Tensor,
        target: Tensor | None,
    ) -> Tensor:
        """
        """

        loss = super().loss(inputs, attention_mask, target)
        e2e_l = self.sigma * self.e2e_loss(inputs, attention_mask, target)

        return loss + e2e_l

    def forward(
        self: Wav2Vec2BertE2E,
        inputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        """

        batch_size = len(inputs)

        hidden_states = super().forward(inputs, attention_mask)

        lengths = attention_mask.sum(dim=-1).to("cpu")

        hidden_states = hidden_states.transpose(0, 1)
        packed_hidden_states = pack_padded_sequence(
            hidden_states, lengths, enforce_sorted=False)

        _, context = self.lstm_encoder(packed_hidden_states)

        logits = self.decode(context)

        return logits

    def decode(
        self: Wav2Vec2BertE2E,
        context: Tensor,
    ) -> Tensor:
        """
        """

        hs, cs = context
        _, batch_size, _ = list(hs.shape)

        head = zeros(batch_size, 1).to(int64)
        head = one_hot(head, num_classes=self.n_classes).to(float32)

        tail = ones(batch_size, 1).to(int64)
        tail = one_hot(tail, num_classes=self.n_classes).to(float32)

        logits = head
        mask = ones(batch_size).to(bool)
        for _ in range(256):
            tmp = logits[mask].transpose(0, 1)
            context = (hs[:, mask, :], cs[:, mask, :])

            tmp = self.pre_linear(tmp)
            tmp, _ = self.lstm_decoder(tmp, context)
            tmp = tmp[-1]
            tmp = self.post_linear(tmp)
            tmp = softmax(tmp, dim=-1)

            tmp = tmp[None, ...].transpose(0, 1)
            clone = tail.clone()
            clone[mask] = tmp
            logits = cat((logits, clone), dim=1)

            idx = logits[:, -1].argmax(dim=-1)

            mask = (idx != 1)
            if mask.any():
                continue
            break

        return logits


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
