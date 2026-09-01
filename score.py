#!/usr/bin/env python3
"""Score ProtAudit embeddings and write a TSV plus score-band plot."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "models" / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Score embeddings with a frozen ProtAudit MLP.")
    parser.add_argument("embeddings", type=Path, help="Directory created by embed.py")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output TSV")
    parser.add_argument("--plot", type=Path, help="Output plot (default: <output stem>_plot.png)")
    parser.add_argument("--model", help="Override model recorded in summary.json")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path, expected):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "row" not in reader.fieldnames:
            raise ValueError(f"Invalid ID table: {path}")
        id_column = "protein_id" if "protein_id" in reader.fieldnames else "canonical_id"
        ids = []
        for expected_row, row in enumerate(reader):
            if int(row["row"]) != expected_row:
                raise ValueError(f"Nonconsecutive row in {path}: expected {expected_row}")
            ids.append(row[id_column])
    if len(ids) != expected:
        raise ValueError(f"ID/embedding count mismatch: {len(ids)} != {expected}")
    return ids


def mlp_class(torch, dimension):
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths = [dimension, dimension // 2, dimension // 4, dimension // 6]
            layers = []
            for incoming, outgoing in zip(widths[:-1], widths[1:]):
                layers.extend(
                    (torch.nn.Linear(incoming, outgoing), torch.nn.LayerNorm(outgoing), torch.nn.ReLU())
                )
            self.features = torch.nn.Sequential(*layers)
            self.classifier = torch.nn.Linear(widths[-1], 1)

        def forward(self, values):
            return self.classifier(self.features(values)).squeeze(1)

    return MLP


def make_plot(scores, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    counts = np.array([(scores >= 0.9).sum(), ((scores >= 0.5) & (scores < 0.9)).sum(), (scores < 0.5).sum()])
    fractions = counts / len(scores)
    labels = (r"$\geq$0.9", "0.5–0.9", "<0.5")
    colors = ("#3F78A0", "#B8D7AA", "#EE999B")

    fig, ax = plt.subplots(figsize=(12, 2.7))
    left = 0.0
    for count, fraction, color in zip(counts, fractions, colors):
        ax.barh(0, fraction, left=left, height=0.62, color=color, edgecolor="none")
        if count:
            text = f"{count:,} ({fraction:.0%})"
            ax.text(left + fraction / 2, 0, text, ha="center", va="center", fontsize=13, fontweight="bold")
        left += fraction
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    fig.suptitle("ProtAudit Scores", x=0.12, y=0.91, ha="left", fontsize=23)
    handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.67, 0.92), ncol=3,
               frameon=False, fontsize=15, handlelength=0.8, handletextpad=0.35, columnspacing=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    summary_path = args.embeddings / "summary.json"
    array_path = args.embeddings / "embeddings.npy"
    ids_path = args.embeddings / "ids.tsv"
    for path in (summary_path, array_path, ids_path, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(MANIFEST.read_text())
    model_name = args.model or summary.get("model")
    if model_name not in manifest["models"]:
        raise ValueError(f"Unknown or missing embedding model: {model_name}")
    if args.model and summary.get("model") and args.model != summary["model"]:
        raise ValueError(
            f"--model {args.model} conflicts with embedding metadata: {summary['model']}"
        )
    info = manifest["models"][model_name]
    if summary.get("model") != model_name and args.model is None:
        raise ValueError("Embedding metadata does not identify a supported model")
    if summary.get("embedding_file_sha256") and sha256(array_path) != summary["embedding_file_sha256"]:
        raise ValueError("Embedding checksum does not match summary.json")

    embeddings = np.load(array_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != info["dimension"]:
        raise ValueError(f"Expected (*, {info['dimension']}) embeddings; found {embeddings.shape}")
    if len(embeddings) == 0:
        raise ValueError("Embedding array contains no proteins")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity")
    ids = read_ids(ids_path, len(embeddings))

    import torch

    checkpoint_path = ROOT / info["checkpoint"]
    if sha256(checkpoint_path) != info["sha256"]:
        raise ValueError(f"Frozen checkpoint checksum mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("dimension") != info["dimension"] or checkpoint.get("layers") != 3:
        raise ValueError("Frozen checkpoint metadata is incompatible")
    model = mlp_class(torch, info["dimension"])().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), args.batch_size):
            values = torch.from_numpy(np.asarray(embeddings[start : start + args.batch_size], dtype=np.float32))
            predictions.append(torch.sigmoid(model(values)).numpy())
    scores = np.concatenate(predictions)
    threshold = float(info["validation_threshold"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("protein_id", "protein_likeness_score", "passes_frozen_threshold", "score_band"))
        for identifier, score in zip(ids, scores):
            band = ">=0.9" if score >= 0.9 else "0.5-0.9" if score >= 0.5 else "<0.5"
            writer.writerow((identifier, f"{score:.8f}", str(bool(score >= threshold)), band))
    plot_path = args.plot or args.output.with_name(f"{args.output.stem}_plot.png")
    make_plot(scores, plot_path)
    print(f"Scored {len(scores):,} proteins with {model_name}")
    print(f"TSV:  {args.output}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
