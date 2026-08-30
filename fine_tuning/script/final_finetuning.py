"""Train a final, deployable fine-tuned checkpoint.

Takes a fold from the dataset (0 by default), merges train
and test and trains on this new split.

Same config surface as `scripts/evaluate.py` -- just:
  - a non-`*_linear_probing` task (this script is fine-tuning only), and
  - a freshly-built train/val split replacing the task's on-disk CV fold JSON

Results land under the usual <probing.results_dir>/<eval.model_name>/<exp_name>/<task_dir>/:
  - the kept checkpoint (`best.ckpt`)
  - `final_split.json` (the exact train/val participant split used)
"""

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger

# Ensure the repo root is on the path when calling from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multimodality.comm_encoder import CoMMEncoder
from src.multimodality.multimodal_wrapper import MultimodalWrapper
from src.multimodality.unimodal_wrapper import UnimodalWrapper
from nidl.utils.weights import Weights
from src.data.dataset import FOMO26Dataset
from src.probing.finetuning import run_one_fold_finetuning
from src.probing.lora import apply_lora
from src.utils import rootutils

rootutils.setup_root(__file__, indicator=".env")


def build_final_split(dataset: FOMO26Dataset, fold: int, use_last_checkpoint: bool) -> list:
    """Merge fold `fold`'s train and test splits into the final training set;
    keep its val split for early-stopping checkpoint selection. `test` reuses
    `val` as a structural placeholder -- run_one_fold_finetuning requires one,
    but there's nothing meaningful left to hold out at this stage.

    If using last checkpoint then all data is used for finetuning because validation
    is ignored.
    """
    if dataset.folds is None:
        raise ValueError("No folds defined on this dataset -- can't build a final split.")
    if fold >= len(dataset.folds):
        raise ValueError(f"Fold {fold} does not exist; only {len(dataset.folds)} available.")
    source = dataset.folds[fold]
    if use_last_checkpoint:
        final_train = source["train"] + source["val"] + source["test"]
        return [{"train": final_train, "val": [], "test": source["val"]}]  # quick and dirty for compatibility, TODO adapt finetuning.py to completely skip test for final_finetuning
    final_train = source["train"] + source["test"]
    return [{"train": final_train, "val": source["val"], "test": source["val"]}]


@hydra.main(config_path="../configs", config_name="finetuning", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.task.probing.finetuning:
        raise ValueError(
            "final_finetuning.py only supports the fine-tuning path -- pass a task whose "
            "`probing.finetuning` is true (e.g. task=CLS002_FOMO26_Infarct, not "
            "task=CLS002_FOMO26_Infarct_linear_probing)."
        )

    # Whether to save the best checkpoint (wrt to val_loss) or the last checkpoint
    # If using the last checkpoint then all data is used for finetuning
    # and validation is disabled.
    use_last_checkpoint = cfg.get('use_last_checkpoint', False)

    # --- Build preprocessing ---
    trainval_transform = None
    test_transform = None
    augmentation_transform = None
    use_augmentation_on_val = False
    target_transform = None
    inverse_target_transform = None

    if cfg.task.probing.get("trainval_transform") is not None:
        trainval_transform = hydra.utils.instantiate(cfg.task.probing.trainval_transform, _convert_="partial")
    if cfg.task.probing.get("test_transform") is not None:
        test_transform = hydra.utils.instantiate(cfg.task.probing.test_transform, _convert_="partial")
    if cfg.task.probing.get("augmentation_transform") is not None:
        augmentation_transform = hydra.utils.instantiate(
            cfg.task.probing.augmentation_transform, _convert_="partial")
        if cfg.task.probing.get("use_augmentation_on_val") is not None:
            use_augmentation_on_val = cfg.task.probing.use_augmentation_on_val
        else:
            raise ValueError(
                "When data agumentation operations are specified, use_augmentation_on_val must be provided"
            )
    if cfg.task.probing.get("target_transform") is not None:
        target_transform = hydra.utils.instantiate(cfg.task.probing.target_transform, _convert_="partial")
        if cfg.task.probing.get("invert_target_transform") is not None:
            inverse_target_transform = hydra.utils.instantiate(cfg.task.probing.invert_target_transform, _convert_="partial")
        else:
            raise ValueError(
                "When a target transform is specified, an inverse must be provided"
            )

    # --- Instantiate model class and encoder (untrained) ---
    encoder_untrained = hydra.utils.instantiate(cfg.eval.encoder)
    if cfg.eval.get('model_class'):
        model_class = hydra.utils.instantiate(cfg.eval.model_class)

    # --- Load pretrained checkpoint ---
    if not cfg.eval.get("checkpoint"):
        raise ValueError(
            "A pretrained checkpoint is required. Set `checkpoint` in the model config."
        )
    ckpt_path = cfg.eval.checkpoint
    weights = Weights(
        name=str(Path(ckpt_path).parent),
        data_dir=str(Path(ckpt_path).parent),
        filepath=Path(ckpt_path).name,
    )
    if ckpt_path.endswith('.ckpt'):
        # pass `model_kwargs` for CoMM
        load_kwargs = {}
        if cfg.eval.get("model_kwargs") is not None:
            load_kwargs = hydra.utils.instantiate(cfg.eval.model_kwargs, _convert_="partial")
        model = weights.load_checkpoint(
            model_class,
            encoder=encoder_untrained,
            logger=None,
            weights_only=False,
            map_location='cpu',
            **load_kwargs)
        # Extract trained encoder
        encoder = getattr(model, cfg.eval.encoder_attr)
    else:
        weights.load_pretrained(encoder_untrained)  # inplace
        encoder = encoder_untrained

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_sd = raw_ckpt.get("state_dict", raw_ckpt)
    prefix = cfg.eval.encoder_attr + "."
    enc_sd = encoder.state_dict()
    missing, mismatched, matched = [], [], 0
    for k, v in enc_sd.items():
        ck = prefix + k
        if ck not in ckpt_sd:
            missing.append(k)
        elif not torch.equal(v.cpu(), ckpt_sd[ck].to(v.dtype).cpu()):
            mismatched.append(k)
        else:
            matched += 1
    print(f"[ckpt] {ckpt_path}")
    print(f"[ckpt] encoder='{cfg.eval.encoder_attr}': "
          f"{matched}/{len(enc_sd)} tensors match checkpoint exactly")
    if missing:
        print(f"[ckpt] WARNING: {len(missing)} encoder keys not found under "
              f"'{prefix}' in checkpoint: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if mismatched:
        print(f"[ckpt] WARNING: {len(mismatched)} encoder tensors differ from "
              f"checkpoint values: {mismatched[:10]}{' ...' if len(mismatched) > 10 else ''}")
    if missing or mismatched:
        raise RuntimeError(
            "Encoder was not fully/correctly loaded from checkpoint — see [ckpt] warnings above."
        )

    # --- Adapt encoder to n_modalities if requested ---
    if cfg.eval.get("comm_fusion", False):
        # `encoder` here is CoMM's MissingModMMFusion.
        mod_slots = cfg.task.get("modality_slots", None)
        if mod_slots is not None:
            mod_slots = list(mod_slots)
        encoder = CoMMEncoder(encoder, mod_slots=mod_slots)
        freeze_backbone = cfg.eval.get("freeze_backbone", False)
        encoder.freeze_backbone(freeze_backbone)
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in encoder.parameters())
        print(f"[extract] CoMM fusion encoder (prototype embedding), "
            f"modality_slots={mod_slots if mod_slots is not None else 'default 0..n-1'}, "
            f"freeze_backbone={freeze_backbone} "
            f"(trainable {n_train:,}/{n_total:,} params)")

    # If segmentation head is defined then multimodality handled directly in the head for other encoders:
    elif (cfg.task.dataset.task_type != "segmentation"
          or cfg.eval.get("segmentation_decoder", None) is None):
        if cfg.eval.get("late_fusion_multimodal", False):
            encoder = UnimodalWrapper(encoder)
        else:
            encoder = MultimodalWrapper(encoder) # the NoFusionEncoder averages embeddings across tokens for a ViT

    # Configure sliding window prediction (optional)
    sliding_window_cfg = None
    sw_cfg = cfg.task.probing.get("sliding_window", None)
    if sw_cfg and sw_cfg.get("enabled", False):
        sliding_window_cfg = {
            "patch_size": tuple(sw_cfg.get("patch_size", cfg.eval.target_shape)),
            "overlap": sw_cfg.get("overlap", 0.5),
            "sw_batch_size": sw_cfg.get('sw_batch_size', 1),
        }
        # If using sliding_window then no resizing of input should be used.
        if test_transform.does_resizing():
            raise ValueError("preprocessing transform has a resizing step but sliding_window is enabled. "
                             "The core improvment of sliding window is to work on full scan resolution. Consider "
                              "setting eval.preprocessing.resizing_method: null.")
            

    # --- Instantiate dataset ---
    dataset_partial = hydra.utils.instantiate(cfg.task.dataset)
    dataset = dataset_partial(trainval_transform=trainval_transform,
                              test_transform=test_transform,
                              augmentation_transform=augmentation_transform,
                              use_augmentation_on_val=use_augmentation_on_val,
                              target_transform=target_transform)

    dataset.folds = build_final_split(dataset, fold=cfg.task.probing.fold, use_last_checkpoint=use_last_checkpoint)
    print(f"[final_finetuning] {dataset.name}: final split -- "
          f"{len(dataset.folds[0]['train'])} train / {len(dataset.folds[0]['val'])} val")

    # --- Output directory (created by Hydra) ---
    results_dir = HydraConfig.get().runtime.output_dir

    with open(os.path.join(results_dir, "final_split.json"), "w") as split_file:
        json.dump(dataset.folds, split_file, indent=2)

    finetuning_kwargs = OmegaConf.to_container(
        cfg.task.probing.finetuning_kwargs, resolve=True
    )
    finetuning_kwargs = {k: v for k, v in finetuning_kwargs.items() if v is not None}
    finetuning_kwargs.setdefault(
        "logger",
        CSVLogger(save_dir=results_dir, name="lightning_logs", version=""),
    )

    finetuning_callbacks = []
    callbacks_cfg = cfg.task.probing.get("finetuning_callbacks")
    if callbacks_cfg:
        for cb_cfg in callbacks_cfg:
            finetuning_callbacks.append(hydra.utils.instantiate(cb_cfg))

    target_shape = (
        tuple(cfg.eval.target_shape)
        if cfg.eval.get("target_shape")
        else (128, 128, 128)
    )

    # If enabled, add LoRA adapter to encoder
    lora_cfg = cfg.eval.get("lora")
    if lora_cfg is not None and lora_cfg.enabled:
        encoder = apply_lora(
            encoder,
            target=lora_cfg.target,
            r=lora_cfg.r,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
        )
        print(
            f"***LoRA Enabled***"
            f"[lora] target={lora_cfg.target} r={lora_cfg.r} "
            f"alpha={lora_cfg.alpha} dropout={lora_cfg.dropout}"
        )

        # If LoRA is enabled, freeze_encoder is ignored
        print(
            "LoRA enable: freeze_encoder=true ignored"
            "base encoder stays frozen via PEFT."
        )
        freeze_encoder = False
    else:
        freeze_encoder = cfg.task.probing.get('freeze_encoder', False)

    final_ckpt_path = run_one_fold_finetuning(
        encoder=encoder,
        dataset=dataset,
        fold=0,  # the single final split built above
        latent_size=cfg.eval.latent_size,
        target_shape=target_shape,
        finetuning_kwargs=finetuning_kwargs,
        finetuning_callbacks=finetuning_callbacks,
        batch_size=cfg.task.probing.batch_size,
        num_workers=cfg.task.probing.num_workers,
        pin_memory=cfg.task.probing.pin_memory,
        results_dir=results_dir,
        random_state=cfg.task.probing.random_state,
        freeze_encoder=freeze_encoder,
        seg_decoder_config=cfg.eval.segmentation_decoder,
        sliding_window=sliding_window_cfg,
        inverse_target_transform=inverse_target_transform,
        keep_checkpoint=True,
        use_last_checkpoint=use_last_checkpoint,
    )

    # The written metrics.tsv comes from the val split reused as "test"
    # Delete this file as it can be misleading.
    misleading_metrics = os.path.join(results_dir, "metrics.tsv")
    if os.path.exists(misleading_metrics):
        os.remove(misleading_metrics)

    print(
        f"[final_finetuning] task={cfg.task.name} done -> "
        f"checkpoint kept at {final_ckpt_path}"
    )


if __name__ == "__main__":
    main()