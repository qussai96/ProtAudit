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
    # Protein-level classifiers were trained with non-overlapping windows.
    # Only the ProtT5 residue localizer uses overlapping contextual windows.
    "prott5": {"dimension": 1024, "window": 1000, "residue_overlap": 128},
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
    parser.add_argument(
        "--skip-residue-embeddings", action="store_true",
        help="Do not retain ProtT5 residue embeddings (disables error localization)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path, rejected=None):
    """Yield usable records and optionally collect rejected-record diagnostics.

    A malformed biological record must not terminate an audit of every other
    protein. Structural FASTA errors (for example sequence before any header)
    remain fatal because record boundaries cannot be recovered safely.
    """
    header, pieces, seen, header_line = None, [], set(), None

    def finish(identifier, sequence, line_number):
        sequence = sequence.upper()
        if sequence.endswith((".", "*")):
            sequence = sequence[:-1]
        reason = None
        if identifier in seen:
            reason = "duplicate_id"
        elif not sequence:
            reason = "empty_sequence"
        else:
            invalid = sorted(set(sequence) - ALLOWED_AA)
            if invalid:
                reason = ("internal_stop" if set(invalid).issubset({".", "*"})
                          else "invalid_residues:" + "".join(invalid))
        if reason:
            if rejected is None:
                raise ValueError(f"Rejected FASTA record {identifier}: {reason}")
            rejected.append({
                "protein_id": identifier,
                "header_line": line_number,
                "reason": reason,
                "sequence_length": len(sequence),
            })
            return None
        seen.add(identifier)
        return identifier, sequence

    with path.open("r", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    record = finish(header, "".join(pieces), header_line)
                    if record is not None:
                        yield record
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA ID at line {line_number}")
                header, pieces, header_line = fields[0], [], line_number
            elif header is None:
                raise ValueError(f"Sequence before first header at line {line_number}")
            else:
                pieces.append(line)
    if header is not None:
        record = finish(header, "".join(pieces), header_line)
        if record is not None:
            yield record


def make_windows(records, size, overlap=0):
    windows = []
    for row, (_, sequence) in enumerate(records):
        if not overlap or len(sequence) <= size:
            starts = range(0, len(sequence), size)
        else:
            stride = size - overlap
            starts = list(range(0, len(sequence) - size + 1, stride))
            final = len(sequence) - size
            if starts[-1] != final:
                starts.append(final)
        for start in starts:
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
            checkpoint = "Rostlab/prot_t5_xl_half_uniref50-enc"
            tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(checkpoint).eval()
        except ImportError as exc:
            raise RuntimeError(
                "ProtT5 dependencies could not be imported. Install requirements.txt. "
                f"Original error: {exc!r}"
            ) from exc
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
                result.append(reps[i, :residues].float().cpu().numpy())
            return result

        return model, encode, {
            "checkpoint": checkpoint,
            "transformers_version": transformers.__version__,
        }

    try:
        from sequence_models.pretrained import load_model_and_alphabet
        model, collater = load_model_and_alphabet("carp_640M")
    except ImportError as exc:
        raise RuntimeError(
            "CARP dependencies could not be imported. Install requirements.txt. "
            f"Original error: {exc!r}"
        ) from exc
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


def cuda_compatibility(torch, device_index=0):
    """Return (usable, reason) for the installed PyTorch binary and GPU."""
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    try:
        major, minor = torch.cuda.get_device_capability(device_index)
        architecture = major * 10 + minor
        compiled = set(torch.cuda.get_arch_list())
    except Exception as exc:
        return False, f"CUDA compatibility check failed: {exc}"
    exact = f"sm_{architecture}"
    ptx = [int(item.split("_")[1]) for item in compiled if item.startswith("compute_")]
    if exact in compiled or any(item <= architecture for item in ptx):
        return True, f"compute capability {major}.{minor} is supported"
    return False, (
        f"GPU compute capability {major}.{minor} is not supported by this PyTorch build "
        f"({', '.join(sorted(compiled)) or 'no CUDA architectures reported'})"
    )


def choose_device(torch, requested, purpose="embedding"):
    usable, reason = cuda_compatibility(torch)
    if requested == "cuda":
        if not usable:
            raise RuntimeError(f"--device cuda requested for {purpose}, but {reason}")
        return torch.device("cuda:0")
    if requested == "auto" and usable:
        return torch.device("cuda:0")
    if requested == "auto" and torch.cuda.is_available():
        print(f"WARNING: {reason}; falling back to CPU for {purpose}.", file=sys.stderr)
    return torch.device("cpu")


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

    device = choose_device(torch, args.device)
    use_cuda = device.type == "cuda"
    rejected = []
    records = list(read_fasta(args.fasta, rejected=rejected))
    if not records:
        raise ValueError("Input FASTA contains no records")
    config = CONFIGS[args.model]
    retain_residues = args.model == "prott5" and not args.skip_residue_embeddings
    # This is the frozen protein-classifier preprocessing for every model and
    # must never depend on whether residue embeddings are retained.
    windows = make_windows(records, config["window"], overlap=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.parent / f".{args.output.name}.partial.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    started = time.time()
    model, encode, backend = load_backend(args.model, torch, device)
    sums = np.zeros((len(records), config["dimension"]), dtype=np.float32)
    weights = np.zeros(len(records), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum([len(sequence) for _, sequence in records]))).astype(np.int64)
    residue_sums = residue_counts = residue_saved = None
    residue_path = offsets_path = None
    long_rows, long_offsets = [], {}
    if retain_residues:
        residue_path = temporary / "residue_embeddings.npy"
        offsets_path = temporary / "residue_offsets.npy"
        residue_saved = np.lib.format.open_memmap(
            residue_path, mode="w+", dtype=np.float16,
            shape=(int(offsets[-1]), config["dimension"]),
        )
        # Float32 accumulation is required only for long proteins whose
        # overlapping localizer windows need averaging—not for the proteome.
        long_rows = [row for row, (_, seq) in enumerate(records) if len(seq) > config["window"]]
        long_offsets = {row: start for row, start in zip(
            long_rows,
            np.cumsum([0] + [len(records[row][1]) for row in long_rows[:-1]]),
        )}
        long_residue_count = sum(len(records[row][1]) for row in long_rows)
        if long_residue_count:
            residue_sums = np.memmap(
                temporary / "residue_sums.tmp", mode="w+", dtype=np.float32,
                shape=(long_residue_count, config["dimension"]),
            )
            residue_counts = np.memmap(
                temporary / "residue_counts.tmp", mode="w+", dtype=np.uint16,
                shape=(long_residue_count,),
            )

    batch_count = 0
    for batch_count, batch in enumerate(
        batches(windows, args.max_batch_tokens, args.max_batch_size), 1
    ):
        vectors = encode_with_oom_split(batch, encode, torch)
        for item, vector in zip(batch, vectors):
            length, row, start, _ = item
            if args.model == "prott5" and vector.shape == (length, config["dimension"]):
                if retain_residues and len(records[row][1]) <= config["window"]:
                    begin = int(offsets[row])
                    residue_saved[begin:begin + length] = vector.astype(np.float16)
                vector = vector.mean(axis=0)
            if vector.shape != (config["dimension"],) or not np.isfinite(vector).all():
                raise RuntimeError(f"Invalid embedding for {records[row][0]}: shape={vector.shape}")
            sums[row] += vector * length
            weights[row] += length
        if batch_count % 100 == 0:
            print(f"Processed {batch_count:,} embedding batches", flush=True)

    expected = np.array([len(sequence) for _, sequence in records])
    if retain_residues:
        residue_windows = make_windows(
            [records[row] for row in long_rows], config["window"], config["residue_overlap"]
        )
        # make_windows indexes its subset; restore original protein rows.
        restored_windows = [
            (length, long_rows[subset_row], start, piece)
            for length, subset_row, start, piece in residue_windows
        ]
        for batch_count, batch in enumerate(
            batches(restored_windows, args.max_batch_tokens, args.max_batch_size), 1
        ):
            vectors = encode_with_oom_split(batch, encode, torch)
            for item, vector in zip(batch, vectors):
                length, row, start, _ = item
                if vector.shape != (length, config["dimension"]) or not np.isfinite(vector).all():
                    raise RuntimeError(
                        f"Invalid residue embedding for {records[row][0]}: shape={vector.shape}"
                    )
                begin = int(long_offsets[row]) + start
                end = begin + length
                residue_sums[begin:end] += vector
                residue_counts[begin:end] += 1
        if long_rows and np.any(residue_counts == 0):
            raise RuntimeError("At least one residue was not embedded")
        for row in long_rows:
            begin, end = int(offsets[row]), int(offsets[row + 1])
            compact_begin = int(long_offsets[row])
            compact_end = compact_begin + (end - begin)
            values = (np.asarray(residue_sums[compact_begin:compact_end]) /
                      np.asarray(residue_counts[compact_begin:compact_end])[:, None])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite residue embedding for {records[row][0]}")
            residue_saved[begin:end] = values.astype(np.float16)
        residue_saved.flush()
        np.save(offsets_path, offsets)
        del residue_saved, residue_sums, residue_counts
        for scratch in (temporary / "residue_sums.tmp", temporary / "residue_counts.tmp"):
            if scratch.exists():
                scratch.unlink()
    if not np.array_equal(weights, expected):
        raise RuntimeError("Not every residue contributed exactly once")
    embeddings = sums / weights[:, None]
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise RuntimeError("Final embeddings failed finite/nonzero validation")

    embedding_path, ids_path = temporary / "embeddings.npy", temporary / "ids.tsv"
    np.save(embedding_path, embeddings.astype(np.float16))
    with ids_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("row", "protein_id"))
        writer.writerows((row, identifier) for row, (identifier, _) in enumerate(records))
    rejected_path = temporary / "rejected_records.tsv"
    with rejected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("protein_id", "header_line", "reason", "sequence_length"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rejected)
    with (temporary / "sequences.faa").open("w", encoding="ascii") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")
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
        "rejected_record_count": len(rejected),
        "rejected_records_file": "rejected_records.tsv",
        "rejected_records_file_sha256": sha256(rejected_path),
        "window_count": len(windows),
        "window_size": config["window"],
        "long_sequence_policy": (
            f"non-overlapping {config['window']}-residue windows; residue-count-weighted mean"
        ),
        "embedding_dimension": config["dimension"],
        "embedding_dtype": "float16",
        "embedding_shape": list(saved.shape),
        "embedding_file_sha256": sha256(embedding_path),
        "ids_file_sha256": sha256(ids_path),
        "sequences_file_sha256": sha256(temporary / "sequences.faa"),
        "residue_embeddings_retained": retain_residues,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if use_cuda else platform.processor(),
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if retain_residues:
        summary.update({
            "residue_embedding_file": "residue_embeddings.npy",
            "residue_embedding_shape": [int(offsets[-1]), config["dimension"]],
            "residue_embedding_dtype": "float16",
            "residue_embedding_file_sha256": sha256(residue_path),
            "residue_offsets_file": "residue_offsets.npy",
            "residue_offsets_file_sha256": sha256(offsets_path),
            "residue_window_overlap": config["residue_overlap"],
            "residue_window_count": len(restored_windows),
            "residue_long_sequence_policy": (
                f"{config['window']}-residue windows with {config['residue_overlap']}-residue "
                "overlap and C-terminal anchoring; overlapping representations averaged"
            ),
        })
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    del saved, model
    if args.output.exists():
        shutil.rmtree(args.output)
    os.replace(temporary, args.output)
    print(f"Saved {len(records):,} {args.model} embeddings to {args.output}")
    if rejected:
        print(f"Skipped {len(rejected):,} rejected FASTA records; see {args.output / 'rejected_records.tsv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
