import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "residualvit", "src"))

from open_clip.factory import create_model

# Absolute path anchored to the MA-LMM project root (two levels up from this file)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "residualVIT", "ViT-L-14", "checkpoints", "epoch_5.pt")


class ResidualViTEncoder(nn.Module):
    """
    Wraps ResidualViT's VisionTransformer so it is a drop-in replacement
    for EVA-ViT inside MA-LMM.

    EVA-ViT returns: Tensor [B, N+1, width]   (CLS + patch tokens, pre-proj)
    ResidualViT returns: dict {'features': [B, output_dim], 'tokens': [B, N, width]}

    This wrapper reconstructs a [B, N+1, width] tensor and exposes num_features=width
    so the Q-Former is sized correctly.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.visual.output_tokens = True  # ensure patch tokens are returned

        # width = raw feature dim before projection (e.g. 1024 for ViT-L-14)
        self.num_features = visual.transformer.width

    def forward(self, x):
        # cast input to match conv1 weight dtype (fp16 or fp32)
        x = x.to(dtype=self.visual.conv1.weight.dtype)
        out = self.visual(x)
        # out['tokens']: [B, N, width]  — patch tokens before projection
        patch_tokens = out["tokens"]

        # Reconstruct a CLS-like token at the same width by averaging patch tokens.
        # This matches what EVA-ViT returns as x[:,0] (CLS at width dim).
        cls_token = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, width]

        return torch.cat([cls_token, patch_tokens], dim=1)  # [B, N+1, width]


def create_residualvit_encoder(precision="fp32"):
    model = create_model(
        "ViT-L-14",
        pretrained=CHECKPOINT_PATH,
        precision=precision,
    )
    return ResidualViTEncoder(model.visual)
