"""
"""

from math import prod

from torch import Tensor
from torch import zeros
from torch import cat
from torch import logsumexp

from torch.nn.functional import cosine_similarity
from torch.nn.functional import cross_entropy

from .internals import norm, guard
from .quantizers import _VectorQuantizer
from .validation import is_infinitesimal


def contrastive_loss(
    hidden_states: Tensor,
    codevectors: Tensor,
    masked_idxs: list[Tensor],
    negative_idxs: list[Tensor],
    tau: float = 0.1
) -> Tensor:
    """
    """

    is_infinitesimal(tau)

    batch_size = len(hidden_states)
    loss = zeros((batch_size, ))

    for i in range(batch_size):

        tmp_idxs = masked_idxs[i].unsqueeze(-1)
        masked_hidden_states = hidden_states[i, tmp_idxs, :]

        tmp_idxs = cat((tmp_idxs, negative_idxs[i]), dim=-1)
        candidates = codevectors[i, tmp_idxs, :]

        similarities = cosine_similarity(
            masked_hidden_states,
            candidates,
            dim=-1
        )
        self_similarities = similarities[:, 0]

        tmp_loss = (self_similarities / tau).exp()
        tmp_loss = tmp_loss / ((similarities / tau).exp().sum(-1))
        loss[i] = - tmp_loss.mean().log()

    return loss.mean()


def mlm_loss(
    hidden_state: Tensor,
    codewords: Tensor,
    masked_idxs: list[Tensor]
) -> Tensor:
    """
    """

    batch_size = len(hidden_state)
    loss = zeros((batch_size, ))

    for i in range(batch_size):

        masked_hidden_state = hidden_state[i, masked_idxs[i], :]
        tmp_codewords = codewords[i, masked_idxs[i], :]
        length, n_groups = list(tmp_codewords.shape)

        masked_hidden_state = masked_hidden_state.view(length, -1, n_groups)
        masked_probs = masked_hidden_state.softmax(-2)

        loss[i] = cross_entropy(masked_probs, tmp_codewords)

    return loss.mean()


def log_likelihood(
    inputs: Tensor,
    attention_mask: Tensor,
    vq: _VectorQuantizer
) -> Tensor:
    """
    """

    logits = vq.logits(inputs, attention_mask)
    log_likelihood = logsumexp(logits, dim=-1).mean(-1)
    log_likelihood = log_likelihood.sum(-1) / attention_mask.sum(-1)

    return log_likelihood.mean()


def entropy(
    inputs: Tensor,
    attention_mask: Tensor,
    vq: _VectorQuantizer
) -> Tensor:
    """
    """

    logits = vq.logits(inputs, attention_mask)
    logits = norm(logits)
    entropy = - (logits.exp() * logits).sum(-1).mean(-1)
    entropy = entropy.sum(-1) / attention_mask.sum(-1)

    return entropy.mean()


def b_diversity(
    inputs: Tensor,
    attention_mask: Tensor,
    vq: _VectorQuantizer
) -> Tensor:
    """
    """

    logits = vq.logits(inputs, attention_mask)
    logits = norm(logits)
    probs = logits[attention_mask].exp()
    avg_probs = guard(probs.mean(0))
    b_diversity = - (avg_probs * avg_probs.log()).sum(-1)

    return b_diversity.mean()


def similarity(vq: _VectorQuantizer) -> Tensor:
    """
    """

    size = vq.codebook_size
    codevectors = vq.codebook.codevectors

    similarity = zeros(size, size)
    for i in range(size):
        similarity[i, :] = cosine_similarity(codevectors[i], codevectors)

    similarity = similarity.triu(diagonal=1)

    return similarity.mean()


def e2e_loss(
    logits: Tensor,
    target: Tensor
) -> Tensor:
    """
    """

    return cross_entropy(logits, target)


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
