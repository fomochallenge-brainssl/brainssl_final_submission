import os
import sys
import glob
import hydra
from omegaconf import DictConfig
from lightning_fabric import seed_everything
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import rootutils
from utils.comm_utils import comm_collate

PRETRAINING_CONF = "pretrain_neurojepa"

@hydra.main(config_path="../configs", config_name=PRETRAINING_CONF, version_base="1.3")
def run_experiment(cfg: DictConfig):
    print("Setup...\n", flush=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    rootutils.setup_root(__file__, indicator=".env")

    seed_everything(cfg.experiment.seed)

    # Build (stochastic, multi-view) transform
    print("Building transform...\n", flush=True)
    transform = None
    if cfg.experiment.get("transform") is not None:
        transform = hydra.utils.instantiate(cfg.experiment.transform, _convert_='partial')

    # Build multimodal datasets (multimodal/n_views come from the dataset config)
    print("Building dataset...\n", flush=True)
    dataset_factory = hydra.utils.instantiate(cfg.dataset)
    training_dataset = dataset_factory(transforms=transform, split='train')
    validation_dataset = dataset_factory(transforms=transform, split='val')

    # Data loaders use the packed multimodal collate
    print("Building dataloader...\n", flush=True)
    pretraining_loader = DataLoader(
        training_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=cfg.experiment.shuffle,
        drop_last=cfg.experiment.drop_last,
        pin_memory=cfg.experiment.pin_memory,
        num_workers=cfg.experiment.num_workers,
        prefetch_factor=cfg.experiment.prefetch_factor,
        collate_fn=comm_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.experiment.pin_memory,
        num_workers=2,
        prefetch_factor=cfg.experiment.prefetch_factor,
        collate_fn=comm_collate,
    )

    # Encoder backbones, checkpoints loaded inside the CoMM estimator:
    print("Building encoder...\n", flush=True)
    encoders = [hydra.utils.instantiate(cfg.encoder)]
    if cfg.get("encoder_dwi") is not None:
        encoders.append(hydra.utils.instantiate(cfg.encoder_dwi))
        print("intialized separate encoder for dwi")
    for encoder in encoders:
        if hasattr(encoder, "embedding"): #since using resnet truncated
            encoder.embedding.requires_grad_(False)

    # Modality slot -> encoder index, following the dataset's channel order
    channels = list(cfg.dataset.channels)
    mod_to_encoder = None
    if len(encoders) > 1:
        dwi_channels = set(cfg.get("dwi_channels") or [])
        mod_to_encoder = [1 if c in dwi_channels else 0 for c in channels]
        if 1 not in mod_to_encoder:
            raise ValueError(
                f"None of the dataset channels {channels} is in "
                f"dwi_channels={sorted(dwi_channels)}: the DWI encoder would "
                f"never be used.")
        print("Modality -> encoder:",
              dict(zip(channels, mod_to_encoder)), "\n", flush=True)

    print("Building callback functions...\n", flush=True)
    checkpointing_callback = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoints_dir,
        monitor='loss/val',
        save_top_k=1,
        save_on_exception=True,
        mode='min',
        save_last='link',
        every_n_epochs=cfg.experiment.checkpointing_epochs
    )

    metrics_logging = WandbLogger(
        name=cfg.experiment.metrics.run_name,
        save_dir=cfg.experiment.metrics.path,
        offline=True,
    )

    print("Building model...\n", flush=True)
    model_factory = hydra.utils.instantiate(cfg.model)
    model = model_factory(
        encoder=encoders,
        num_modalities=len(channels),
        mod_to_encoder=mod_to_encoder,
        callbacks=checkpointing_callback,
        random_state=cfg.experiment.seed,
        accelerator=cfg.experiment.accelerator,
        logger=metrics_logging,
        log_every_n_steps=cfg.experiment.metrics.logging_interval,
    )

    warm_start_ckpt = cfg.experiment.get("warm_start_ckpt", None)
    latest_checkpoint = None
    if warm_start_ckpt:
        print(f"Warm-starting weights from {warm_start_ckpt}...\n", flush=True)
        state_dict = torch.load(warm_start_ckpt, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  missing keys: {missing}\n  unexpected/mismatched keys: {unexpected}", flush=True)
    else:
        # restart from latest checkpoint, if it exists
        print("Loading checkpoint file...\n", flush=True)
        checkpoint_files = glob.glob(os.path.join(cfg.experiment.checkpoints_dir, "last.ckpt"))
        latest_checkpoint = checkpoint_files[0] if checkpoint_files else None
        print('Resume from checkpoint:', latest_checkpoint)

    print("Training model...\n", flush=True)
    model.fit(pretraining_loader, validation_loader, latest_checkpoint)
    print("Training completed!!!\n", flush=True)


if __name__ == "__main__":
    run_experiment()