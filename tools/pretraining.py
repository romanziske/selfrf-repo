import os
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from selfrf.pretraining.config import TrainingConfig, parse_training_config, print_config
from selfrf.pretraining.factories import build_dataloader, build_ssl_model
from selfrf.pretraining.utils.callbacks import ModelAndBackboneCheckpoint


def train(config: TrainingConfig):

    # Print confirmation
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        print(f"Current CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    datamodule = build_dataloader(config)
    datamodule.prepare_data()
    datamodule.setup()

    # if not config.online_linear_eval:
    # datamodule.val_dataloader = None

    ssl_model = build_ssl_model(config)

    # Configure TensorBoardLogger
    logger = TensorBoardLogger(
        os.path.join(config.training_path, config.ssl_model.value)
    )

    # Configure ModelCheckpoint
    checkpoint_callback = ModelAndBackboneCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            f"{config.ssl_model.name}"
            f"-{config.backbone.value}"
            f"-{config.dataset.value}"
            f"-{'spec' if config.spectrogram else 'iq'}"
            f"-e{{epoch:d}}"
            f"-b{config.batch_size}"
            f"-loss{{train_loss:.9f}}"
        ),
        save_top_k=100,
        verbose=True,
        save_last=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        precision="16-mixed",
        accelerator=config.device.type,
        callbacks=[checkpoint_callback],
        logger=logger,
        accumulate_grad_batches=8,
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
    )

    trainer.fit(
        model=ssl_model,
        datamodule=datamodule,
        ckpt_path=config.resume_from_checkpoint,
    )


if __name__ == '__main__':
    load_dotenv()
    config = parse_training_config()
    print_config(config)
    train(config)
