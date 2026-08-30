"""
    This file contains the code for the segmentation decoder of the UNETR model.
    Adapted from: https://arxiv.org/abs/2103.10504 (Hatamizadeh et al., WACV 2022)
"""
import torch
from torch import nn


def _norm(features):
    # Instance norm rather than batch norm to work with small batches
    return nn.InstanceNorm3d(features, affine=True)


"""
    Stores the output of the transformer layer it is attached to into
    store[key] to keep the module savable
"""
class ActivationSaver:
    def __init__(self, store, key):
        self.store = store
        self.key = key

    def __call__(self, model, input, output):
        # MoE ViT blocks return (x, moe_scores)
        if isinstance(output, tuple):
            output = output[0]
        self.store[self.key] = output


"""
    Two 3x3x3 convolutions with a skip connection
"""
class ResidualConvUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv1 = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm1 = _norm(out_features)
        self.conv2 = nn.Conv3d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(out_features)
        self.activation = nn.LeakyReLU(0.01, inplace=True)

        # Project the input when the skip connection changes the channel count
        if in_features != out_features:
            self.skip = nn.Sequential(
                nn.Conv3d(in_features, out_features, kernel_size=1, bias=False),
                _norm(out_features),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # In shape: (B, in_features, D0, D1, D2)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.activation(out + self.skip(x))
        # Out shape: (B, out_features, D0, D1, D2)


"""
    Projects a transformer layer representation back to image space and
    upsamples it by a factor of 2 ** (upsampling_steps + 1), so that skip
    connections taken from earlier layers land at higher resolutions
"""
class ProjectionUpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, upsampling_steps):
        super().__init__()

        self.projection = nn.ConvTranspose3d(in_features, out_features, kernel_size=2, stride=2)
        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(out_features, out_features, kernel_size=2, stride=2),
                ResidualConvUnit(out_features, out_features),
            )
            for _ in range(upsampling_steps)
        ])

    def forward(self, x):
        # In shape: (B, in_features, N_patch0, N_patch1, N_patch2)
        x = self.projection(x)
        for block in self.upsampling:
            x = block(x)
        return x
        # Out shape: (B, out_features, 2 ** (steps + 1) * N_patch0, ...)


"""
    One step of the decoder: upsample the partial representation by 2,
    concatenate the matching skip connection and fuse them
"""
class UpsamplingFusionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_features, out_features, kernel_size=2, stride=2)
        # The skip connection always carries out_features channels
        self.fuse = ResidualConvUnit(2 * out_features, out_features)

    def forward(self, x, skip):
        # In shape: (B, in_features, D0, D1, D2) and (B, out_features, 2 * D0, 2 * D1, 2 * D2)
        x = self.upsample(x)
        return self.fuse(torch.cat([x, skip], dim=1))
        # Out shape: (B, out_features, 2 * D0, 2 * D1, 2 * D2)


"""
    UNETR segmentation decoder on top of a pre-trained 3D ViT encoder.

    encoder:
        Pre-trained ViT encoder. Must expose ``blocks``, ``num_prefix_tokens``
        and ``forward_features``.

    num_classes:
        Number of classes of the segmentation problem

    original_image_size / patch_size:
        Used to fold the token sequence back onto the patch grid

    latent_space_size:
        Encoder's embedding size

    features_count:
        Width of the decoder. Channel counts are features_count * (1, 2, 4, 8)
        Reduce this to avoid overfitting if needed

    hooks:
        Transformer layers the three skip connections are taken at. The
        bottleneck always comes from the encoder's final output.
"""
class UNETR(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes = 2,
        original_image_size = (128, 128, 128),
        patch_size = (16, 16, 16),
        latent_space_size = 768,
        features_count = 16,
        hooks = (2, 5, 8),
    ):
        super().__init__()

        for size, patch in zip(original_image_size, patch_size):
            assert size % patch == 0, "image size must be a multiple of the patch size"

        self.encoder = encoder
        self.start_index = encoder.num_prefix_tokens
        self.original_image_size = tuple(original_image_size)
        self.grid_shape = torch.Size([
            original_image_size[0] // patch_size[0],
            original_image_size[1] // patch_size[1],
            original_image_size[2] // patch_size[2],
        ])

        # Store the representations of the layers the skip connections start from
        self.latent_reps = {}
        for skip_index, layer in enumerate(hooks):
            encoder.blocks[layer].register_forward_hook(
                ActivationSaver(self.latent_reps, skip_index)
            )

        features = features_count

        # Skip connections, from the highest resolution down to the coarsest
        self.skip1 = ResidualConvUnit(1, features)                                  # full resolution
        self.skip2 = ProjectionUpsamplingBlock(latent_space_size, features * 2, 2)  # 1/2
        self.skip3 = ProjectionUpsamplingBlock(latent_space_size, features * 4, 1)  # 1/4
        self.skip4 = ProjectionUpsamplingBlock(latent_space_size, features * 8, 0)  # 1/8

        # Decoder, from the bottleneck back up to the original resolution
        self.up4 = UpsamplingFusionBlock(latent_space_size, features * 8)
        self.up3 = UpsamplingFusionBlock(features * 8, features * 4)
        self.up2 = UpsamplingFusionBlock(features * 4, features * 2)
        self.up1 = UpsamplingFusionBlock(features * 2, features)

        self.class_predictions_head = nn.Conv3d(features, num_classes, kernel_size=1)

    def forward(self, x):
        # In shape: (B, C, original_image_size[0], original_image_size[1], original_image_size[2])
        B, C = x.shape[0], x.shape[1]

        # Encode different modalities as independent samples
        x = x.reshape(B * C, 1, *self.original_image_size)

        # Bottleneck; the hooks fill self.latent_reps as a side effect
        bottleneck = self.to_patch_grid(self.encoder.forward_features(x))
        # (B * C, latent_space_size, N_patch0, N_patch1, N_patch2)

        # Read the skip representations and drop the references the hooks hold,
        # so that activations neither survive the step nor reach the checkpoint
        skip2, skip3, skip4 = [
            self.to_patch_grid(self.latent_reps.pop(index)) for index in range(3)
        ]

        partial_rep = self.up4(bottleneck, self.skip4(skip4))
        partial_rep = self.up3(partial_rep, self.skip3(skip3))
        partial_rep = self.up2(partial_rep, self.skip2(skip2))
        partial_rep = self.up1(partial_rep, self.skip1(x))
        # (B * C, features_count, original_image_size[0], original_image_size[1], original_image_size[2])

        # Fuse modalities
        partial_rep = partial_rep.reshape(B, C, *partial_rep.shape[1:]).mean(dim=1)

        return self.class_predictions_head(partial_rep)
        # Out shape: (B, num_classes, original_image_size[0], original_image_size[1], original_image_size[2])

    def to_patch_grid(self, tokens):
        # (B, N_token, latent_space_size) -> (B, latent_space_size, N_patch0, N_patch1, N_patch2)
        tokens = tokens[:, self.start_index:]
        return tokens.transpose(1, 2).unflatten(2, self.grid_shape)

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class CoMMUNetR(nn.Module):
    """
    CoMM variant of UNETR above, with the modality axis collapsed
    at the bottleneck instead of after the decoder.

    encoder:
        A pretrained ``MissingModMMFusion``, or the ``CoMMEncoder`` wrapper
        around one. The fusion must hold a single shared trunk, and the 
        trunk must be a transformer exposing ``blocks``, i.e. the adapter 
        is a tokenizer such as``IdentityAdapter``, not a patch embedder.

    num_classes:
        Number of classes of the segmentation problem

    original_image_size / patch_size:
        Used to fold the token sequence back onto the patch grid

    latent_space_size:
        Token width the fusion transformer works in

    features_count:
        Width of the decoder. Channel counts are features_count * (1, 2, 4, 8);
        the main knob to trade capacity against overfitting on small datasets.

    hooks:
        Transformer layers the three skip connections are taken at. The
        bottleneck always comes from the fused final tokens.

    NOTE: currently ONLY supports a single shared encoder and single adapter!!!
"""
    def __init__(
        self,
        encoder: nn.Module,
        num_classes = 2,
        original_image_size = (128, 128, 128),
        patch_size = (16, 16, 16),
        latent_space_size = 768,
        features_count = 16,
        hooks = (2, 5, 8),
    ):
        super().__init__()

        for size, patch in zip(original_image_size, patch_size):
            assert size % patch == 0, "image size must be a multiple of the patch size"

        if len(hooks) != 3:
            raise ValueError(
                f"UNETR has exactly three skip connections, got {len(hooks)} hooks."
            )

        # Accept either the CoMMEncoder wrapper or the raw fusion module
        self.encoder = getattr(encoder, "fusion", encoder)
        self.mod_slots = getattr(encoder, "mod_slots", None)

        # check for single shared encoder and single adapter!!
        if len(self.encoder.encoders) > 1:
            raise ValueError(
                "CoMMUNetR reads its skip connections off one shared trunk, but "
                f"this fusion holds {len(self.encoder.encoders)}. Multi-encoder "
                "routing is not yet supported."
            )

        if len(self.encoder.adapters) > 1:
            raise ValueError(
                f"This fusion was pretrained with per_modality_adapter=True "
                f"({len(self.encoder.adapters)} adapters), but CoMMUNetR "
                "doesn't yet support multi-adapter routing. "
            )

        self.original_image_size = tuple(original_image_size)
        self.patch_size = tuple(patch_size)
        self.grid_shape = torch.Size([
            original_image_size[0] // patch_size[0],
            original_image_size[1] // patch_size[1],
            original_image_size[2] // patch_size[2],
        ])
        # Trunks that emit prefix/cls tokens: those are dropped, only patch
        # tokens fold back onto the grid
        self.start_index = getattr(self.backbone, "num_prefix_tokens", 0)

        fusion_features = self.encoder.fusion_transformer.width
        if fusion_features != latent_space_size:
            raise ValueError(
                f"latent_space_size={latent_space_size} but the pretrained fusion "
                f"transformer works in width {fusion_features}. The bottleneck is "
                "the fused token grid, so the two have to agree."
            )
        # The skips are read before the adapter, so they carry the trunk's own
        # width, which a tokenizing adapter can change
        trunk_features = getattr(self.backbone, "embed_dim", latent_space_size)

        # Store the representations of the layers the skip connections start from
        self.latent_reps = {}
        for skip_index, layer in enumerate(hooks):
            self.backbone.blocks[layer].register_forward_hook(
                ActivationSaver(self.latent_reps, skip_index)
            )
        self.num_skips = len(hooks)

        features = features_count

        # Skip connections, from the highest resolution down to the coarsest.
        self.skip1 = ResidualConvUnit(1, features)                             # full resolution
        self.skip2 = ProjectionUpsamplingBlock(trunk_features, features * 2, 2)  # 1/2
        self.skip3 = ProjectionUpsamplingBlock(trunk_features, features * 4, 1)  # 1/4
        self.skip4 = ProjectionUpsamplingBlock(trunk_features, features * 8, 0)  # 1/8

        # Decoder, from the fused bottleneck back up to the original resolution
        self.up4 = UpsamplingFusionBlock(fusion_features, features * 8)
        self.up3 = UpsamplingFusionBlock(features * 8, features * 4)
        self.up2 = UpsamplingFusionBlock(features * 4, features * 2)
        self.up1 = UpsamplingFusionBlock(features * 2, features)

        self.class_predictions_head = nn.Conv3d(features, num_classes, kernel_size=1)

    @property
    def backbone(self):
        return self.encoder.encoders[0]

    def forward(self, x):
        # In shape: (B, n, original_image_size[0], original_image_size[1], original_image_size[2])
        B, n = x.shape[0], x.shape[1]

        # Encode each modality of each subject independently
        x = x.reshape(B * n, 1, *self.original_image_size)

        # Row b * n + m of `x` is subject b, downstream channel m
        sample_idx = torch.arange(B, device=x.device).repeat_interleave(n)
        mod_idx = self._modality_slots(n, x.device).repeat(B)

        # Trunk; the hooks are read inside, so the skips come back in scan order
        tokens, skips = self._encode(x)   # (B * n, N_token, E)
        self._check_token_count(tokens.shape[1])

        # Bottleneck: CoMM's pretrained attention over the modality tokens,
        # in place of UNETR's mean over decoded feature maps
        grid, absent = self.encoder.scatter(
            self._tokenize(tokens), sample_idx, mod_idx, B
        )
        z = self._fuse_tokens(grid, absent)                 # (B, M, N_token, E)

        # Mean over the modalities this subject actually has
        present = (~absent).to(z.dtype)[..., None, None]    # (B, M, 1, 1)
        z = (z * present).sum(dim=1) / present.sum(dim=1).clamp(min=1)   # (B, N_token, E)

        bottleneck = self.to_patch_grid(z)
        # (B, latent_space_size, N_patch0, N_patch1, N_patch2)

        # Skips are per-modality trunk features, average them to one per subject
        # to match bottleneck
        skip2, skip3, skip4 = [
            self.to_patch_grid(self._reduce(skip, B, n)) for skip in skips
        ]
        # Same at voxel resolution, where the only feature is the volume itself
        volume = self._reduce(x, B, n)

        partial_rep = self.up4(bottleneck, self.skip4(skip4))
        partial_rep = self.up3(partial_rep, self.skip3(skip3))
        partial_rep = self.up2(partial_rep, self.skip2(skip2))
        partial_rep = self.up1(partial_rep, self.skip1(volume))
        # (B, features_count, original_image_size[0], original_image_size[1], original_image_size[2])

        return self.class_predictions_head(partial_rep)
        # Out shape: (B, num_classes, original_image_size[0], original_image_size[1], original_image_size[2])

    def to_patch_grid(self, tokens):
        # (B, N_token, E) -> (B, E, N_patch0, N_patch1, N_patch2)
        tokens = tokens[:, self.start_index:]
        return tokens.transpose(1, 2).unflatten(2, self.grid_shape)

    @staticmethod
    def _reduce(features, B, n):
        """Average a per-scan tensor down to one entry per subject."""
        return features.reshape(B, n, *features.shape[1:]).mean(dim=1)

    def _encode(self, x):
        """The trunk's final tokens, plus what its hooks captured on the way.

        The freeze flag is honoured as ``MissingModMMFusion.encode`` does: a
        frozen trunk runs under ``no_grad``, so the skips are constants exactly
        as the bottleneck is.

        :param x: volumes (N, 1, H, W, D).
        :return: ``(tokens, skips)``, all (N, N_token, E) in scan order.
        """
        if getattr(self.encoder, "freeze_encoder", False):
            with torch.no_grad():
                tokens = self.backbone(x)
        else:
            tokens = self.backbone(x)

        if isinstance(tokens, tuple):
            tokens = tokens[0]

        # Read the skip representations and drop the references the hooks hold,
        # so that activations neither survive the step nor reach the checkpoint
        skips = [self.latent_reps.pop(index) for index in range(self.num_skips)]

        return tokens, skips

    def _tokenize(self, tokens):
        """Run the fusion's shared adapter over the trunk tokens."""
        return self.encoder.adapters[0](tokens)   # (N, T, E)

    def _modality_slots(self, n, device):
        """Pretraining slot index of each of the n downstream channels."""
        num_mod = self.encoder.num_modalities
        slots = list(range(n)) if self.mod_slots is None else self.mod_slots

        if len(slots) != n:
            raise ValueError(
                f"mod_slots has {len(slots)} entries but the task provides "
                f"{n} modalities; they must match (one slot per channel)."
            )

        if max(slots) >= num_mod or min(slots) < 0:
            raise ValueError(
                f"Modality slots {slots} are out of range for a CoMM fusion "
                f"pretrained with num_modalities={num_mod} (valid: 0..{num_mod - 1})."
            )

        return torch.tensor(slots, device=device, dtype=torch.long)

    def _check_token_count(self, num_tokens):
        """The token sequence has to reshape onto the patch grid exactly."""
        expected = self.grid_shape.numel() + self.start_index
        if num_tokens != expected:
            raise ValueError(
                f"The trunk returns {num_tokens} tokens per scan, but "
                f"original_image_size={self.original_image_size} with "
                f"patch_size={self.patch_size} implies a {tuple(self.grid_shape)} "
                f"grid ({expected} tokens, {self.start_index} of them prefix "
                "tokens). A trunk that tokenizes the volume differently needs "
                "those settings adjusted."
            )

    def _fuse_tokens(self, grid, absent):
        """CoMM's fusion transformer, keeping every token: (B, M, T, D) in, out.
        """
        fusion_transformer = self.encoder.fusion_transformer
        B, M, T, D = grid.shape

        # one sequence of M*T tokens per subject, the batch axis keeps subjects apart
        x = grid.reshape(B, M * T, D)
        mask = absent.unsqueeze(-1).expand(B, M, T).reshape(B, M * T)

        if fusion_transformer.pool == "cls":
            # kept for attention, so attention matches pretraining
            cls_token = fusion_transformer.cls_token.expand(B, 1, D)
            x = torch.cat([cls_token, x], dim=1)
            mask = torch.cat([mask.new_zeros(B, 1), mask], dim=1)

        key_padding_mask = torch.zeros_like(x[..., 0]).masked_fill(
            mask, float("-inf")
        )

        for layer in fusion_transformer.resblocks:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = fusion_transformer.norm(x)

        if fusion_transformer.pool == "cls":
            x = x[:, 1:]

        return x.reshape(B, M, T, D)

    def freeze_encoder(self):
        """Freeze the whole fusion encoder: trunks, adapters and attention."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.freeze_encoder = True

    def unfreeze_encoder(self):
        """Unfreeze the whole fusion encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.freeze_encoder = False