import torch
from lightning_fabric import seed_everything
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
from utils import rootutils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import glob
from utils import utils

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def run_experiment(cfg: DictConfig):
    # setup project root and load .env file
    print(f"[{utils.timestamp()}] Setup...\n", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    rootutils.setup_root(__file__, indicator=".env")

    # Set random seed
    seed_everything(cfg.experiment.seed)

    # Build transform
    print(f"[{utils.timestamp()}] Building transform...\n", flush=True)
    transform = None
    if cfg.experiment.get("transform") is not None:
        transform = hydra.utils.instantiate(cfg.experiment.transform, _convert_='partial')

    # Build dataset
    print(f"[{utils.timestamp()}] Building dataset...\n", flush=True)
    dataset_factory =hydra.utils.instantiate(cfg.dataset)
    training_dataset = dataset_factory(
        transforms=transform,
        split='train'
    )
    validation_dataset = dataset_factory(
        transforms=transform,
        split='val'
    )

    # Instantiate data loaders
    print(f"[{utils.timestamp()}] Building dataloader...\n", flush=True)
    pretraining_loader = DataLoader(
        training_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=cfg.experiment.shuffle,
        drop_last=cfg.experiment.drop_last,
        pin_memory=cfg.experiment.pin_memory,
        num_workers=cfg.experiment.num_workers,
        prefetch_factor=cfg.experiment.prefetch_factor,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.experiment.pin_memory,
        num_workers=cfg.experiment.num_workers,
        prefetch_factor=cfg.experiment.prefetch_factor,
    )

    print(f"[{utils.timestamp()}] Building encoder...\n", flush=True)
    encoder = hydra.utils.instantiate(cfg.encoder)

    # Callback functions to save best and last checkpoints
    print(f"[{utils.timestamp()}] Building callback functions...\n", flush=True)
    best_checkpoint = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoints_dir,
        monitor='loss/val',
        save_top_k = 1,
        save_on_exception = True,
        mode='min',
        save_last=False, # Set to False to avoid confusion with last_model.ckpt
        every_n_epochs=cfg.experiment.checkpointing_epochs
    )

    last_checkpoint = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoints_dir,
        monitor=None, # Saves model for last checkpoint
        filename="last_model",
        save_on_exception=True,
        save_last=False,
        every_n_epochs=cfg.experiment.checkpointing_epochs,      
    )
    checkpointing_callback = [best_checkpoint, last_checkpoint]

    # Locally log training loss using wandb
    metrics_logging = [
        WandbLogger(
            name = cfg.experiment.metrics.run_name,
            save_dir = cfg.experiment.metrics.path,
            offline = True
        ),
        CSVLogger(
            save_dir=cfg.experiment.output_dir,
            name='csv/',
            prefix=''
        )]

    # Instantiate model
    print(f"[{utils.timestamp()}] Building model...\n", flush=True)
    model_factory = hydra.utils.instantiate(cfg.model)
    model = model_factory(
        encoder=encoder,
        callbacks=checkpointing_callback,
        random_state=cfg.experiment.seed,
        accelerator=cfg.experiment.accelerator,
        logger=metrics_logging,
    )

    warm_start_ckpt = cfg.experiment.get("warm_start_ckpt", None)
    if warm_start_ckpt:
        print(f"[{utils.timestamp()}] Warm-starting weights from {warm_start_ckpt}...\n", flush=True)
        state_dict = torch.load(warm_start_ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  missing keys: {missing}\n  unexpected/mismatched keys: {unexpected}", flush=True)
    else:
        # restart from latest checkpoint, if it exists
        print(f"[{utils.timestamp()}] Loading checkpoint file...\n", flush=True)
        checkpoint_files = glob.glob(os.path.join(cfg.experiment.checkpoints_dir, "last.ckpt"))

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
    else:
        latest_checkpoint = None
    
    # Launch model training
    torch.set_float32_matmul_precision('medium')
    print(f"[{utils.timestamp()}] Training model...\n", flush=True)
    model.fit(pretraining_loader, validation_loader, latest_checkpoint)
    print(f"[{utils.timestamp()}] Training completed!!!\n", flush=True)

if __name__ == "__main__":
    run_experiment()