"""LoRA adapter injection for ViT-based encoders (DINO / VisionTransformer3D)."""
from typing import Literal

from peft import LoraConfig, get_peft_model
from torch import nn

# Layer of the transformer architecture where the LoRA can be attached
LoRATarget = Literal["attention", "mlp"]
LORA_TARGET_MODULES: dict[LoRATarget, list[str]] = {
    "attention": ["qkv", "proj"],
    "mlp": ["fc1", "fc2"],
}

def apply_lora(
    encoder: nn.Module,
    target: LoRATarget,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> nn.Module:
    """Wrap a ViT encoder with PEFT LoRA adapters on the selected layers."""
    if target not in LORA_TARGET_MODULES:
        raise ValueError(
            f"LoRA target must be one of {list(LORA_TARGET_MODULES)}, got {target!r}."
        )

    # Setting up the LoRA layer
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES[target],
        lora_dropout=dropout,
        bias="none",
    )

    # Add the LoRA in the desired spot and return the resulting encoder
    encoder = get_peft_model(encoder, config)
    encoder.print_trainable_parameters()

    return encoder