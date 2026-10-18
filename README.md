<p align="center">
  <img src="assets/ProtAudit_logo.jpg" alt="ProtAudit logo" width="280">
</p>

# ProtAudit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qussai96/ProtAudit/blob/main/ProtAudit_Colab.ipynb)

ProtAudit uses a sequential audit workflow. First, the frozen binary model calls each
sequence **positive** (protein-like) or **negative**. For ProtT5 binary-negative
sequences, a separate frozen diagnostic model then reports probabilities for
four simulated error classes: chimeric fusion, splice artifact, terminal
abnormality, and Spurious ORFs. These diagnostic
classes describe sequence resemblance and are not confirmed causal annotation
errors. For negatives assigned to the chimeric, splice-artifact, or
terminal-abnormal classes, the final residue-level model also nominates the
highest-probability error location using a broad-class-conditioned dilated
residual convolutional network. Spurious ORFs are not localized because the
entire translated ORF may be unrelated to a genuine protein.

The repository includes frozen binary MLP classifiers for ProtT5 (default),
ESM-2 8M, ESM-2 650M, and CARP 640M. Conditional diagnostic probabilities are
currently available for ProtT5 only, and residue localization requires ProtT5.

For a small FASTA containing up to 100 proteins, open the Colab notebook using
the badge above, select a GPU runtime, upload the FASTA, and run all cells. The
notebook downloads a complete TSV, a localization-only TSV, and the summary
plot. Mean-pooled and residue-level embeddings remain in the runtime and can
also be downloaded using the notebook's optional embeddings download. Use the
local installation below for larger files.

## Installation

Python 3.10+ and a CUDA GPU are recommended. Model weights are downloaded by
their upstream packages on first use.

```bash
git clone https://github.com/qussai96/ProtAudit.git
cd ProtAudit
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
```

## Running

Embed a protein FASTA (ProtT5 is used unless `--model` is supplied), then score
the embeddings:

```bash
python embed.py proteins.faa --output embeddings
python score.py embeddings --output results/protaudit_scores.tsv
```

To use another frozen model, add one of `--model esm2_320`,
`--model esm2_1280`, or `--model carp` to the embedding command. The scoring
command reads the selected model from `embeddings/summary.json` automatically.
Long proteins are embedded without truncation using model-safe windows and a
residue-weighted mean. ProtT5 additionally retains one 1,024-dimensional
float16 vector per residue. Long ProtT5 sequences use 1,000-residue windows
with 128-residue overlap; representations for overlapping residues are
averaged before whole-protein mean pooling and localization.

## Output

`embed.py` writes `embeddings.npy`, `ids.tsv`, `sequences.faa`, and
provenance/checksums in `summary.json`. For ProtT5 it also writes
`residue_embeddings.npy` and `residue_offsets.npy`. The residue store requires
approximately 2 KiB per amino acid because it is saved as float16.

`score.py` writes:

- `protaudit_scores.tsv`: protein ID, binary call, protein-likeness score,
  frozen-threshold call, score band, and (for ProtT5 negatives) the predicted
  diagnostic class, its top probability, and all four class probabilities.
- `protaudit_scores_localization.tsv`: one row per localizable negative with
  the predicted broad class, protein length, one-based peak residue number,
  amino acid, and localization probability.
- `protaudit_scores_plot.png`: positive-versus-negative counts followed by the
  diagnostic-class composition of binary-negative sequences.

The main score table also contains `localization_status`,
`localized_residue_number`, `localized_amino_acid`, and
`localization_probability`. Localization fields are intentionally blank for
positive proteins and Spurious-ORF negatives.

The frozen decision threshold is model-specific and was selected on the
validation species. Diagnostic probabilities are conditional on a sequence
receiving a binary-negative call; they should not be interpreted as biological
proof of a particular annotation error. Likewise, a localization peak is a
candidate site for inspection, not proof of causality or a proposed annotation
repair.

## Citation

If you use ProtAudit, please cite:

> Abbas, Q. et al. (2026). *ProtAudit: protein sequence auditing with protein
> language model embeddings.* Manuscript in preparation.
