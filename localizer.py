#!/usr/bin/env python3
"""Final hard-class-conditioned ProtAudit residue localizer."""

from pathlib import Path

import numpy as np


CLASS_TO_INDEX = {"chimeric": 0, "splice_artifact": 1, "terminal_abnormal": 2}


def model_class(torch):
    class ResidualConvBlock(torch.nn.Module):
        def __init__(self, channels, kernel, dilation, dropout):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.block = torch.nn.Sequential(
                torch.nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation),
                torch.nn.GELU(), torch.nn.Dropout(dropout),
            )
            self.norm = torch.nn.LayerNorm(channels)

        def forward(self, values):
            return self.norm((values + self.block(values)).transpose(1, 2)).transpose(1, 2)

    class CategoryConditionedLocalizer(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            projection = int(config["projection_dimension"])
            category = int(config["category_embedding_dimension"])
            self.input_norm = torch.nn.LayerNorm(1024)
            self.projection = torch.nn.Linear(1024, projection)
            self.category_embedding = torch.nn.Embedding(3, category)
            channels = projection + category
            self.blocks = torch.nn.ModuleList([
                ResidualConvBlock(
                    channels, int(config["convolution_kernel_size"]), int(dilation),
                    float(config["dropout"]),
                )
                for dilation in config["convolution_dilations"]
            ])
            self.output = torch.nn.Conv1d(channels, 1, 1)

        def forward(self, embeddings, categories):
            values = self.projection(self.input_norm(embeddings))
            condition = self.category_embedding(categories)[:, None, :].expand(
                -1, values.shape[1], -1
            )
            values = torch.cat([values, condition], dim=-1).transpose(1, 2)
            for block in self.blocks:
                values = block(values)
            return self.output(values).squeeze(1)

    return CategoryConditionedLocalizer


def load_localizer(torch, checkpoint_path: Path, expected_sha256: str, sha256, device):
    if sha256(checkpoint_path) != expected_sha256:
        raise ValueError(f"Frozen localizer checksum mismatch: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"state_dict", "config", "epoch"}
    if not required.issubset(state):
        raise ValueError("Frozen localizer checkpoint is incomplete")
    model = model_class(torch)(state["config"])
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device), state


def localize_rows(torch, model, residue_embeddings, offsets, rows, classes, device):
    """Return {row: (zero-based peak, probability)} for localizable negatives."""
    output = {}
    with torch.inference_mode():
        for row, error_class in zip(rows, classes):
            start, end = int(offsets[row]), int(offsets[row + 1])
            if end <= start:
                raise ValueError(f"Empty residue embedding for row {row}")
            values = torch.from_numpy(
                np.array(residue_embeddings[start:end], dtype=np.float32, copy=True)
            )[None].to(device)
            category = torch.tensor([CLASS_TO_INDEX[error_class]], dtype=torch.long, device=device)
            probabilities = torch.sigmoid(model(values, category))[0]
            peak = int(torch.argmax(probabilities).item())
            output[row] = (peak, float(probabilities[peak].item()))
    return output
