"""Implementation of additional modules for the VLA's vision transformer."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class FiLMedVisionTransformerBlock(nn.Module):
    """
    Wrapper for ViT blocks that adds components to implement FiLM language conditioning.

    Modulates visual feature embeddings via
        x = (1 + gamma) * x + beta,
    where x is visual feature and gamma and beta are learned projections of the average language embedding.
    """

    def __init__(self, block, vision_dim: int, llm_dim: int):
        super().__init__()
        self.block = block
        self.scale = nn.Linear(llm_dim, vision_dim)
        self.shift = nn.Linear(llm_dim, vision_dim)

    def forward(self, x, average_language_embedding):
        gamma = self.scale(average_language_embedding)
        beta = self.shift(average_language_embedding)

        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        return x


class NullVisionTransformerBlockWrapper(nn.Module):
    """Null wrapper for ViT blocks that doesn't do anything; just calls the original block's forward function."""

    def __init__(self, block, **kwargs):
        super().__init__()
        self.block = block

    def forward(self, x, average_language_embedding=None):
        return self.block(x)


class FiLMedVisionTransformer(VisionTransformer):
    """Vision Transformer with FiLM conditioning via language embeddings."""

    def forward_features(self, x, average_language_embedding=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            if isinstance(block, (FiLMedVisionTransformerBlock, NullVisionTransformerBlockWrapper)):
                x = block(x, average_language_embedding)
            else:
                x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x, average_language_embedding=None):
        x = self.forward_features(x, average_language_embedding)
        return x


def unpack_tuple(fn):
    """Helper to unpack tuple output from timm's get_intermediate_layers."""
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result
    return wrapper


class FiLMedPrismaticVisionBackbone(nn.Module):
    """
    Wraps PrismaticVisionBackbone with FiLM conditioning.
    Replaces ViT blocks with FiLMed versions that accept language embeddings.
    """

    def __init__(self, vision_backbone, llm_dim: int):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        # Wrap the ViT blocks with FiLM
        self._wrap_vit(self.vision_backbone.featurizer)
        if self.vision_backbone.use_fused_vision_backbone:
            self._wrap_vit(self.vision_backbone.fused_featurizer)

    def _wrap_vit(self, vit) -> None:
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.num_features, llm_dim=self.llm_dim)
            )
        vit.blocks = nn.Sequential(*block_wrappers)
        vit.__class__ = FiLMedVisionTransformer
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, n={len(vit.blocks) - 2}))

    def get_num_patches(self) -> int:
        return self.vision_backbone.get_num_patches()

    def get_num_images_in_input(self) -> int:
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM language conditioning.

        Args:
            pixel_values: (B, C, H, W) input image pixels.
            language_embeddings: (B, seq_len, llm_dim) language embeddings for task description.
        """
        average_language_embedding = language_embeddings.mean(dim=1)

        if self.get_num_images_in_input() == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values, average_language_embedding)

            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img, average_language_embedding)
            patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)
            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            images = torch.split(pixel_values, [6] * self.get_num_images_in_input(), dim=1)

            all_patches = []
            for img in images:
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)
                patches = self.vision_backbone.featurizer(img_regular, average_language_embedding)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            return torch.cat(all_patches, dim=1)
