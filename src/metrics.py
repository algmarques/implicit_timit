"""
"""

from collections.abc import Callable

from torch import inference_mode
from torch import float32, int64
from torch import zeros, ones
from torch import cat
from torch import bincount, where

from torch.nn import Module
from torch.nn.functional import cross_entropy, one_hot

from .utils import Arg
from .dataset import WaveDataset


@inference_mode()
def get_contrastive_loss(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    contrastive_loss = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        c_loss = model.contrastive_loss(inputs, attention_mask)
        contrastive_loss += len(attention_mask) * float(c_loss)
        weight += len(attention_mask)
    contrastive_loss /= weight

    return contrastive_loss


@inference_mode()
def get_mlm_loss(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    mlm_loss = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        tmp = model.mlm_loss(inputs, attention_mask)
        mlm_loss += len(attention_mask) * float(tmp)
        weight += len(attention_mask)
    mlm_loss /= weight

    return mlm_loss


@inference_mode()
def get_log_likelihood(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    log_likelihood = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        ll_loss = model.log_likelihood(inputs, attention_mask)
        log_likelihood += len(attention_mask) * float(ll_loss)
        weight += len(attention_mask)
    log_likelihood /= weight

    return log_likelihood


@inference_mode()
def get_entropy(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    entropy = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        e_loss = model.entropy(inputs, attention_mask)
        entropy += len(attention_mask) * float(e_loss)
        weight += len(attention_mask)
    entropy /= weight

    return entropy


@inference_mode()
def get_codebook_diversity(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    size = model.vq.codebook_size
    counts = zeros(size)
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        projected_inputs = model.feature_projector(inputs, attention_mask)
        logits = model.vq.logits(projected_inputs, attention_mask)
        codewords = logits[attention_mask].argmax(-1)
        counts += bincount(codewords.flatten(), minlength=size)
    counts = counts[counts != 0.0]
    counts /= counts.sum()
    c_diversity = - (counts * counts.log()).sum()

    return float(c_diversity)


@inference_mode()
def get_n_unique_codewords(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    codewords = []
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        projected_inputs = model.feature_projector(inputs, attention_mask)
        logits = model.vq.logits(projected_inputs, attention_mask)
        codewords += [logits[attention_mask].argmax(-1)]
    codewords = cat(codewords)

    for i in range(model.vq.n_groups):
        codewords[:, i] *= (model.vq.codebook_size) ** i
    codewords = codewords.sum(-1)
    n_unique_codewords = len(codewords.unique())

    return float(n_unique_codewords)


@inference_mode()
def get_diversity(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 512

    codewords = []
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        projected_inputs = model.feature_projector(inputs, attention_mask)
        logits = model.vq.logits(projected_inputs, attention_mask)
        codewords += [logits[attention_mask].argmax(-1)]
    codewords = cat(codewords)

    for i in range(model.vq.n_groups):
        codewords[:, i] *= (model.vq.codebook_size) ** i
    codewords = codewords.sum(-1)

    _, counts = codewords.unique(return_counts=True)
    counts = counts / counts.sum()
    diversity = - (counts * counts.log()).sum()

    return float(diversity)


@inference_mode()
def get_similarity(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:

    model.eval()

    similarity = model.similarity()

    return float(similarity)


@inference_mode()
def get_e2e_loss(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:
    """
    """

    model.eval()

    batch_size = 64

    e2e_loss = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]
        target = instance["target"]

        batch_e2e_loss = model.e2e_loss(inputs, attention_mask, target)
        e2e_loss += len(attention_mask) * float(batch_e2e_loss)
        weight += len(attention_mask)
    e2e_loss /= weight

    return e2e_loss


@inference_mode()
def get_error_rate(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:

    model.eval()

    batch_size = 64

    err_rt = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]
        target = instance["target"]

        batch_size, t_seq_len = list(target.shape)
        logits = model(inputs, attention_mask)
        pred = logits.argmax(dim=-1)
        _, p_seq_len = list(pred.shape)

        tmp = ones(batch_size, abs(t_seq_len - p_seq_len)).to(int64)
        if p_seq_len < t_seq_len:
            pred = cat((pred, tmp), dim=1)
        if p_seq_len > t_seq_len:
            target = cat((target, tmp), dim=1)

        tmp = ones(batch_size, 1).to(bool)
        mask = (target != 1)
        mask = cat((tmp, mask[:, 0: -1]), dim=1)

        target = where(mask, target, 0)
        pred = where(mask, pred, 0)

        err = (target != pred)
        err = err.sum(-1) / mask.sum(-1)
        err = err.mean()

        err_rt += len(attention_mask) * float(err)
        weight += len(attention_mask)
    err_rt /= weight

    return err_rt


@inference_mode()
def get_perplexity(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:

    model.eval()

    batch_size = 64
    n_classes = kwargs["n_classes"]

    prplxt = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]
        target = instance["target"]

        _, t_seq_len = list(target.shape)
        logits = model(inputs, attention_mask)
        _, l_seq_len, _ = list(logits.shape)

        tmp = ones(batch_size, abs(t_seq_len - l_seq_len)).to(int64)
        if l_seq_len < t_seq_len:
            tmp = one_hot(tmp, num_classes=n_classes).to(float32)
            logits = cat((logits, tmp), dim=1)
        if l_seq_len > t_seq_len:
            target = cat((target, tmp), dim=1)

        tmp = ones(batch_size, 1).to(bool)
        mask = (target != 1)
        mask = cat((tmp, mask[:, 0: -1]), dim=1)

        target = target[mask]
        logits = logits[mask]

        batch_prplxt = cross_entropy(logits, target)

        prplxt += len(attention_mask) * float(batch_prplxt)
        weight += len(attention_mask)
    prplxt /= weight

    return prplxt


@inference_mode()
def get_edit_distance(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> float:

    model.eval()

    batch_size = 64

    edt_dist = 0.0
    weight = 0
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]
        target = instance["target"]

        _, t_seq_len = list(target.shape)
        logits = model(inputs, attention_mask)
        pred = logits.argmax(dim=-1)
        _, p_seq_len = list(pred.shape)

        tmp = ones(batch_size, abs(t_seq_len - p_seq_len)).to(int64)
        if p_seq_len < t_seq_len:
            pred = cat((pred, tmp), dim=1)
        if p_seq_len > t_seq_len:
            target = cat((target, tmp), dim=1)

        tmp = ones(batch_size, 1).to(bool)
        mask = (target != 1)
        mask = cat((tmp, mask[:, 0: -1]), dim=1)

        target = target[mask]
        neg = pred[~mask]
        pred = pred[mask]

        dist = (target != pred).to(float32).sum()
        dist += (neg != 1).to(float32).sum()

        edt_dist += float(dist)
        weight += len(attention_mask)
    edt_dist /= weight

    return edt_dist


def get_metrics(**kwargs: Arg) -> dict[str, Callable]:
    """
    """

    encoding_metrics = {
        "contrastive_loss": get_contrastive_loss,
        "log_likelihood": get_log_likelihood,
        "entropy": get_entropy,
        "codebook_diversity": get_codebook_diversity,
        "diversity": get_diversity,
        "n_unique_codewords": get_n_unique_codewords,
        "similarity": get_similarity
    }

    decoding_metrics = {
        "e2e_loss": get_e2e_loss,
        "error_rate": get_error_rate,
        "perplexity": get_perplexity,
        "edit_distance": get_edit_distance
    }

    if kwargs["ft"]:
        return decoding_metrics

    metrics = encoding_metrics

    if kwargs["zeta"]:
        metrics |= {"mlm_loss": get_mlm_loss}

    if kwargs["sigma"]:
        metrics |= decoding_metrics

    return metrics


def aggregate(values: dict[str, float], **kwargs: Arg) -> dict[str, float]:
    """
    """

    other = dict()
    for key, value in values.items():
        if kwargs["ft"] and key.endswith("_e2e_loss"):
            kind = key.removesuffix("_e2e_loss")
            key = kind + "_loss"
            other[key] = value

        if key.endswith("_contrastive_loss"):
            kind = key.removesuffix("_contrastive_loss")
            key = kind + "_loss"
            other[key] = value

            if kwargs["zeta"]:
                other[key] += values[kind + "_mlm_loss"]
            if kwargs["sigma"]:
                other[key] += values[kind + "_e2e_loss"]

    return values | other


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
