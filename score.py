#!/usr/bin/env python3
"""Run binary ProtAudit scoring and conditional negative diagnostics."""

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


def diagnostic_mlp_class(torch, dimension, class_count):
    class DiagnosticMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths = [dimension, dimension // 2, dimension // 4, dimension // 6]
            layers = []
            for incoming, outgoing in zip(widths[:-1], widths[1:]):
                layers.extend(
                    (torch.nn.Linear(incoming, outgoing), torch.nn.LayerNorm(outgoing), torch.nn.ReLU())
                )
            self.features = torch.nn.Sequential(*layers)
            self.classifier = torch.nn.Linear(widths[-1], class_count)

        def forward(self, values):
            return self.classifier(self.features(values))

    return DiagnosticMLP


def make_plot(scores, threshold, diagnostic_calls, diagnostic_classes, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    positive = int((scores >= threshold).sum())
    binary_counts = np.array([positive, len(scores) - positive])
    binary_labels = ("Positive", "Negative")
    binary_colors = ("#3F78A0", "#EE999B")
    has_diagnostics = diagnostic_calls is not None
    fig, axes = plt.subplots(2 if has_diagnostics else 1, 1,
                             figsize=(12, 5.5 if has_diagnostics else 2.7))
    axes = np.atleast_1d(axes)

    def stacked_bar(ax, counts, labels, colors, title):
        total = counts.sum()
        left = 0.0
        for count, label, color in zip(counts, labels, colors):
            fraction = count / total if total else 0
            ax.barh(0, fraction, left=left, height=0.62, color=color, edgecolor="none")
            if count:
                ax.text(left + fraction / 2, 0, f"{count:,} ({fraction:.0%})",
                        ha="center", va="center", fontsize=11, fontweight="bold")
            left += fraction
        ax.set(xlim=(0, 1), ylim=(-0.55, 0.55), title=title)
        ax.axis("off")
        ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
                  loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(5, len(labels)),
                  frameon=False, fontsize=9)

    stacked_bar(axes[0], binary_counts, binary_labels, binary_colors,
                "Stage 1: binary ProtAudit call")
    if has_diagnostics:
        diagnostic_counts = np.array([(diagnostic_calls == name).sum() for name in diagnostic_classes])
        labels = tuple(name.replace("_", " ") for name in diagnostic_classes)
        colors = ("#4C78A8", "#F28E2B", "#E15759", "#59A14F", "#B07AA1")
        stacked_bar(axes[1], diagnostic_counts, labels, colors,
                    "Stage 2: diagnostic class among binary negatives")
    fig.suptitle("ProtAudit two-stage results", fontsize=18, y=1.01)
    fig.tight_layout()
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

    diagnostic_classes = None
    diagnostic_probabilities = None
    diagnostic_calls = None
    diagnostic_confidences = None
    negative = scores < threshold
    if model_name == "prott5":
        diagnostic_info = manifest["diagnostic_model"]
        if diagnostic_info["embedding_model"] != model_name or diagnostic_info["dimension"] != info["dimension"]:
            raise ValueError("Diagnostic-model metadata is incompatible with ProtT5")
        diagnostic_path = ROOT / diagnostic_info["checkpoint"]
        if sha256(diagnostic_path) != diagnostic_info["sha256"]:
            raise ValueError(f"Frozen diagnostic checkpoint checksum mismatch: {diagnostic_path}")
        diagnostic_state = torch.load(diagnostic_path, map_location="cpu", weights_only=False)
        diagnostic_classes = list(diagnostic_info["classes"])
        if (diagnostic_state.get("dimension") != info["dimension"]
                or diagnostic_state.get("layers") != 3
                or list(diagnostic_state.get("classes", [])) != diagnostic_classes):
            raise ValueError("Frozen diagnostic checkpoint metadata is incompatible")
        diagnostic_model = diagnostic_mlp_class(torch, info["dimension"], len(diagnostic_classes))().eval()
        diagnostic_model.load_state_dict(diagnostic_state["model_state_dict"])
        chunks = []
        negative_rows = np.flatnonzero(negative)
        with torch.inference_mode():
            for start in range(0, len(negative_rows), args.batch_size):
                rows = negative_rows[start : start + args.batch_size]
                values = torch.from_numpy(np.asarray(
                    embeddings[rows], dtype=np.float32
                ))
                chunks.append(torch.softmax(diagnostic_model(values), dim=1).numpy())
        diagnostic_probabilities = (np.concatenate(chunks) if chunks else
                                    np.empty((0, len(diagnostic_classes)), dtype=np.float32))
        codes = diagnostic_probabilities.argmax(axis=1) if len(diagnostic_probabilities) else np.array([], dtype=int)
        diagnostic_calls = np.asarray(diagnostic_classes)[codes]
        diagnostic_confidences = (diagnostic_probabilities[np.arange(len(codes)), codes]
                                  if len(codes) else np.array([], dtype=np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["protein_id", "binary_call", "protein_likeness_score",
                  "passes_frozen_threshold", "score_band", "diagnostic_class",
                  "diagnostic_class_probability"]
        if diagnostic_classes:
            header.extend(f"probability_{name}" for name in diagnostic_classes)
        writer.writerow(header)
        negative_row = 0
        for identifier, score in zip(ids, scores):
            band = ">=0.9" if score >= 0.9 else "0.5-0.9" if score >= 0.5 else "<0.5"
            is_positive = bool(score >= threshold)
            row = [identifier, "positive" if is_positive else "negative", f"{score:.8f}",
                   str(is_positive), band, "", ""]
            if diagnostic_classes:
                row.extend([""] * len(diagnostic_classes))
                if not is_positive:
                    row[5] = diagnostic_calls[negative_row]
                    row[6] = f"{diagnostic_confidences[negative_row]:.8f}"
                    row[7:] = [f"{value:.8f}" for value in diagnostic_probabilities[negative_row]]
                    negative_row += 1
            writer.writerow(row)
    plot_path = args.plot or args.output.with_name(f"{args.output.stem}_plot.png")
    make_plot(scores, threshold, diagnostic_calls, diagnostic_classes, plot_path)
    print(f"Scored {len(scores):,} proteins with {model_name}")
    print(f"Binary calls: {int((~negative).sum()):,} positive; {int(negative.sum()):,} negative")
    if diagnostic_classes:
        print(f"Assigned conditional diagnostic probabilities to {int(negative.sum()):,} negatives")
    else:
        print("Conditional diagnostics are available only for ProtT5 embeddings")
    print(f"TSV:  {args.output}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
