"""
"""

from __future__ import annotations

from typing import Self, Any
from math import sqrt

from torch import Tensor
from torch import zeros, arange
from torch import matmul, einsum
from torch import clamp, softmax

from torch.nn import Module, Linear, Dropout, Embedding

from ..utils import Arg


class SelfAttention(Module):
    """
    """

    _required = {
        "hidden_size",
        "output_hidden_size",
        "n_attention_heads",
        "attention_dropout",
        "position_embeddings_type",
        "left_max_position_embeddings",
        "right_max_position_embeddings"
    }

    def __init__(self: Self, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.n_heads = self.n_attention_heads
        self.head_size = self.hidden_size // self.n_heads

        self.linear_q = Linear(self.hidden_size, self.hidden_size)
        self.linear_k = Linear(self.hidden_size, self.hidden_size)
        self.linear_v = Linear(self.hidden_size, self.hidden_size)
        self.linear_out = Linear(self.hidden_size, self.hidden_size)

        self.dropout = Dropout(self.attention_dropout)

        if self.position_embeddings_type == "relative_key":
            n_positions = self.left_max_position_embeddings \
                + self.right_max_position_embeddings + 1
            self.distance_embedding = Embedding(n_positions, self.head_size)

    def forward(
        self: Self,
        hidden_state: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, ...]:
        """
        """

        batch_size, sequence_length, hidden_size = hidden_state.size()
        query_key_states = hidden_state
        value_states = hidden_state

        query = self.linear_q(query_key_states)
        query = query.view(batch_size, -1, self.n_heads, self.head_size)
        query = query.transpose(1, 2)

        key = self.linear_k(query_key_states)
        key = key.view(batch_size, -1, self.n_heads, self.head_size)
        key = key.transpose(1, 2)

        value = self.linear_v(value_states)
        value = value.view(batch_size, -1, self.n_heads, self.head_size)
        value = value.transpose(1, 2)

        scores = matmul(query, key.transpose(-2, -1))
        scores /= sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = arange(query_length)
            position_ids_r = arange(key_length)
            distance = position_ids_r.view(-1, 1) - position_ids_l.view(-1, 1)
            distance = clamp(
                distance,
                - self.left_max_position_embeddings,
                self.right_max_position_embeddings
            )

            positional_embedding = self.distance_embedding(
                distance + self.left_max_position_embeddings
            )
            positional_embedding = positional_embedding.to(dtype=query.dtype)

            relative_position_attn_weights = einsum(
                "bhld,lrd->bhlr",
                query,
                positional_embedding
            )
            relative_position_attn_weights /= sqrt(self.head_size)
            scores += relative_position_attn_weights

        if attention_mask is not None:
            scores += attention_mask[..., None, None]\
                .transpose(-2, -3).expand_as(scores)

        probs = softmax(scores, dim=-1)
        probs = self.dropout(probs)

        hidden_state = matmul(probs, value)

        hidden_state = hidden_state.transpose(1, 2)
        hidden_state = hidden_state.reshape(
            batch_size,
            -1,
            self.n_heads * self.head_size
        )
        hidden_state = self.linear_out(hidden_state)

        return hidden_state, probs


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
