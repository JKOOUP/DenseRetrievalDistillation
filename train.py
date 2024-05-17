import sys

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from training import LightningModel, configure_callbacks, set_seed, configure_wandb_logger


def main(config: DictConfig) -> None:
    set_seed(config.training.seed)

    model: LightningModel = LightningModel(config)

    logger: WandbLogger = configure_wandb_logger(config)
    logger.watch(model)

    trainer: Trainer = Trainer(
        accelerator="gpu" if config.training.n_gpus > 0 else "cpu",
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=configure_callbacks(config),
        devices=config.training.n_gpus,
        num_sanity_val_steps=0,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=config.training.epochs,
        num_nodes=config.training.n_nodes,
        precision=16,
        strategy=config.training.strategy,
        val_check_interval=config.training.val_check_interval,
        enable_checkpointing=True,
    )

    if config.validate: 
        trainer.validate(model)
        return

    if config.training.use_teacher_embeddings_cache:
        trainer.test(model)

    if config.training.resume_from_checkpoint:
        trainer.fit(model, ckpt_path=config.training.resume_from_checkpoint)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    with open(sys.argv[1]) as config_path:
        config: DictConfig = OmegaConf.load(config_path)
    main(config)
