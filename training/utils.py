import torch
import random
import numpy as np
import typing as tp

from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def configure_callbacks(config: DictConfig) -> tp.List[tp.Any]:
    lr_monitor: LearningRateMonitor = LearningRateMonitor()

    checkpoint_metric_name: str = "_".join(config.data.checkpoint_metric.split("/"))
    model_checkpoint = ModelCheckpoint(
        dirpath=config.training.save_checkpoint_path,
        filename="epoch={epoch}_step={step}_" + checkpoint_metric_name + "={" + config.data.checkpoint_metric + ":.3f}",
        monitor=config.data.checkpoint_metric,
        mode="max",
        save_on_train_epoch_end=False,
        save_last=True,
        auto_insert_metric_name=False,
    )

    model_checkpoint.CHECKPOINT_JOIN_CHAR = "_"
    return [lr_monitor, model_checkpoint]


def configure_wandb_logger(config: DictConfig) -> WandbLogger:
    return WandbLogger(
        name=config.exp_name,
        save_dir=config.training.logs_path,
        project=config.wandb_project
    )