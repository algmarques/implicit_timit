"""
"""

from pathlib import Path

from torch import device as Device
from torch.backends.mps import is_available as is_mps_available
from torch.mps import set_per_process_memory_fraction
from torch import set_default_device
from torch import set_default_dtype, float32

from .utils import Arg, process_kwargs, load_checkpoint
from .run import Run
from .dataset import WaveDataset, get_processor
from .model import get_model
from .schedulers import get_optimizer, get_scheduler, Annealer, Teacher
from .agnostics import train_epoch, evaluate, compute
from .recycle import is_recyclable, recycle


def fork(runs_dir_path: Path, data_dir_path: Path, **kwargs: Arg) -> None:
    """
    """

    train_dir_path = data_dir_path / "train"
    dev_dir_path = data_dir_path / "dev"
    test_dir_path = data_dir_path / "test"

    kwargs = process_kwargs(data_dir_path, **kwargs)

    run = Run(runs_dir_path, **kwargs)

    device = Device("mps" if is_mps_available() else "cpu")
    set_per_process_memory_fraction(1.0)
    set_default_device(device)

    set_default_dtype(float32)

    train_ds = WaveDataset(train_dir_path)
    dev_ds = WaveDataset(dev_dir_path)
    test_ds = WaveDataset(test_dir_path)
    ds_dict = {"train": train_ds, "dev": dev_ds, "test": test_ds}

    model = get_model(**kwargs)
    #if kwargs["ft"]:
    #    pth = runs_dir_path.parent / "pre_trained"
    #    if kwargs["vq"] == "linear":
    #        pth = pth / "linear.pth"
    #    if kwargs["vq"] == "gaussian":
    #        pth = pth / "gaussian.pth"
    #    load_checkpoint(pth, model.encoder, None, None)

    processor = get_processor(**kwargs)

    optimizer = get_optimizer(model, **kwargs)
    scheduler = get_scheduler(optimizer, **kwargs)
    if not kwargs["ft"]:
        annealer = Annealer(model, **kwargs)
    teacher = Teacher(model, **kwargs)

    metrics = dict()
    artifacts = dict()

    if not run.load(model, optimizer, scheduler, metrics, artifacts):
        metrics = evaluate(model, processor, ds_dict, **kwargs)
        artifacts = compute(model, processor, ds_dict, **kwargs)
        run.save(model, optimizer, scheduler, metrics, artifacts)

    if not run.finished():
        epoch = run.epoch
        if is_recyclable(**kwargs):
            recycle(model, processor, train_ds, **kwargs)
        train_epoch(model, processor, optimizer, train_ds, **kwargs)
        scheduler.step()
        if not kwargs["ft"]:
            annealer.step(epoch)
        teacher.step(epoch)
        if epoch == 1 or not epoch % kwargs["skip"]:
            metrics = evaluate(model, processor, ds_dict, **kwargs)
            artifacts = compute(model, processor, ds_dict, **kwargs)
        run.save(model, optimizer, scheduler, metrics, artifacts)

    print(run.finished())
