#!/usr/bin/env python3
"""Create one whole-protein embedding per FASTA record for ProtAudit."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "prott5": {"dimension": 1024, "window": 1000},
    "esm2_1280": {"dimension": 1280, "window": 1022, "layer": 33},
    "esm2_320": {"dimension": 320, "window": 1022, "layer": 6},
    "carp": {"dimension": 1280, "window": 1024, "layer": 56},
}
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed a protein FASTA with ProtT5, ESM-2, or CARP."
    )
    parser.add_argument("fasta", type=Path, help="Input protein FASTA")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", choices=CONFIGS, default="prott5", help="PLM (default: prott5)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-batch-tokens", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path):
    header, pieces, seen = None, [], set()
    with path.open("r", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield validate_record(header, "".join(pieces), seen)
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA ID at line {line_number}")
                header, pieces = fields[0], []
            elif header is None:
                raise ValueError(f"Sequence before first header at line {line_number}")
            else:
                pieces.append(line)
    if header is not None:
        yield validate_record(header, "".join(pieces), seen)


def validate_record(identifier, sequence, seen):
    sequence = sequence.upper()
    if identifier in seen:
        raise ValueError(f"Duplicate FASTA ID: {identifier}")
    seen.add(identifier)
    if not sequence:
        raise ValueError(f"Empty sequence: {identifier}")
    invalid = sorted(set(sequence) - ALLOWED_AA)
    if invalid:
        raise ValueError(f"Invalid residues for {identifier}: {''.join(invalid)}")
    return identifier, sequence


def make_windows(records, size):
    windows = []
    for row, (_, sequence) in enumerate(records):
        for start in range(0, len(sequence), size):
            piece = sequence[start : start + size]
            windows.append((len(piece), row, start, piece))
    windows.sort(key=lambda item: (item[0], item[1], item[2]))
    return windows


def batches(windows, max_tokens, max_size):
    batch, longest = [], 0
    for item in windows:
        proposed = max(longest, item[0])
        if batch and (len(batch) >= max_size or proposed * (len(batch) + 1) > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, item[0])
    if batch:
        yield batch


def amp(torch, device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def load_backend(name, torch, device):
    config = CONFIGS[name]
    if name.startswith("esm2_"):
        try:
            import esm
        except ImportError as exc:
            raise RuntimeError("ESM-2 requires: pip install fair-esm") from exc
        if name == "esm2_1280":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            checkpoint = "esm2_t33_650M_UR50D"
        else:
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            checkpoint = "esm2_t6_8M_UR50D"
        model = model.eval().to(device)
        converter = alphabet.get_batch_converter()

        def encode(items):
            _, _, tokens = converter([(str(i), item[3]) for i, item in enumerate(items)])
            tokens = tokens.to(device)
            with torch.inference_mode(), amp(torch, device):
                output = model(tokens, repr_layers=[config["layer"]], return_contacts=False)
            reps = output["representations"][config["layer"]]
            lengths = (tokens != alphabet.padding_idx).sum(1).tolist()
            return [
                reps[i, 1 : length - 1].float().mean(0).cpu().numpy()
                for i, length in enumerate(lengths)
            ]

        return model, encode, {"checkpoint": checkpoint, "package": "fair-esm"}

    if name == "prott5":
        try:
            import transformers
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError("ProtT5 requires: pip install transformers sentencepiece") from exc
        checkpoint = "Rostlab/prot_t5_xl_half_uniref50-enc"
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(checkpoint).eval()
        if device.type == "cpu":
            model = model.float()
        model = model.to(device)

        def encode(items):
            tokenized = tokenizer(
                [" ".join(item[3]) for item in items],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            with torch.inference_mode(), amp(torch, device):
                reps = model(**tokenized).last_hidden_state
            result = []
            for i, item in enumerate(items):
                residues = item[0]
                active = int(tokenized["attention_mask"][i].sum())
                if active != residues + 1:
                    raise RuntimeError(
                        f"ProtT5 token/residue mismatch: {active - 1} tokens for {residues} residues"
                    )
                result.append(reps[i, :residues].float().mean(0).cpu().numpy())
            return result

        return model, encode, {
            "checkpoint": checkpoint,
            "transformers_version": transformers.__version__,
        }

    try:
        from sequence_models.pretrained import load_model_and_alphabet
    except ImportError as exc:
        raise RuntimeError(
            "CARP requires: pip install git+https://github.com/microsoft/protein-sequence-models.git"
        ) from exc
    model, collater = load_model_and_alphabet("carp_640M")
    model = model.eval().to(device)

    def encode(items):
        tokens = collater([(item[3],) for item in items])[0].to(device)
        with torch.inference_mode(), amp(torch, device):
            reps = model(tokens, repr_layers=[config["layer"]])["representations"][config["layer"]]
        return [reps[i, : item[0]].float().mean(0).cpu().numpy() for i, item in enumerate(items)]

    return model, encode, {"checkpoint": "carp_640M", "representation_layer": config["layer"]}


def encode_with_oom_split(items, encode, torch):
    try:
        return encode(items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            raise RuntimeError(f"CUDA OOM for one {items[0][0]}-residue window")
        middle = len(items) // 2
        return encode_with_oom_split(items[:middle], encode, torch) + encode_with_oom_split(
            items[middle:], encode, torch
        )


def main():
    args = parse_args()
    if args.max_batch_tokens < 1 or args.max_batch_size < 1:
        raise ValueError("Batch limits must be positive")
    if not args.fasta.is_file():
        raise FileNotFoundError(args.fasta)
    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {args.output}")
    if args.output.is_dir() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output} (use --overwrite)")

    import torch

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    records = list(read_fasta(args.fasta))
    if not records:
        raise ValueError("Input FASTA contains no records")
    config = CONFIGS[args.model]
    windows = make_windows(records, config["window"])
    started = time.time()
    model, encode, backend = load_backend(args.model, torch, device)
    sums = np.zeros((len(records), config["dimension"]), dtype=np.float32)
    weights = np.zeros(len(records), dtype=np.int64)

    batch_count = 0
    for batch_count, batch in enumerate(
        batches(windows, args.max_batch_tokens, args.max_batch_size), 1
    ):
        vectors = encode_with_oom_split(batch, encode, torch)
        for item, vector in zip(batch, vectors):
            length, row, _, _ = item
            if vector.shape != (config["dimension"],) or not np.isfinite(vector).all():
                raise RuntimeError(f"Invalid embedding for {records[row][0]}: shape={vector.shape}")
            sums[row] += vector * length
            weights[row] += length
        if batch_count % 100 == 0:
            print(f"Processed {sum(weights):,} residues", flush=True)

    expected = np.array([len(sequence) for _, sequence in records])
    if not np.array_equal(weights, expected):
        raise RuntimeError("Not every residue contributed exactly once")
    embeddings = sums / weights[:, None]
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise RuntimeError("Final embeddings failed finite/nonzero validation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.parent / f".{args.output.name}.partial.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    embedding_path, ids_path = temporary / "embeddings.npy", temporary / "ids.tsv"
    np.save(embedding_path, embeddings.astype(np.float16))
    with ids_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("row", "protein_id"))
        writer.writerows((row, identifier) for row, (identifier, _) in enumerate(records))
    saved = np.load(embedding_path, mmap_mode="r")
    if saved.shape != embeddings.shape or not np.isfinite(saved).all():
        raise RuntimeError("Saved embedding array failed validation")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "model": args.model,
        "backend": backend,
        "input_fasta": str(args.fasta.resolve()),
        "input_fasta_sha256": sha256(args.fasta),
        "record_count": len(records),
        "window_count": len(windows),
        "window_size": config["window"],
        "long_sequence_policy": "non-overlapping windows; residue-count-weighted mean",
        "embedding_dimension": config["dimension"],
        "embedding_dtype": "float16",
        "embedding_shape": list(saved.shape),
        "embedding_file_sha256": sha256(embedding_path),
        "ids_file_sha256": sha256(ids_path),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if use_cuda else platform.processor(),
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    del saved, model
    if args.output.exists():
        shutil.rmtree(args.output)
    os.replace(temporary, args.output)
    print(f"Saved {len(records):,} {args.model} embeddings to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
