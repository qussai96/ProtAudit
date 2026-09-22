# ProtAudit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qussai96/ProtAudit/blob/main/ProtAudit_Colab.ipynb)

ProtAudit uses a two-stage workflow. First, the frozen binary model calls each
sequence **positive** (protein-like) or **negative**. For ProtT5 binary-negative
sequences, a separate frozen diagnostic model then reports probabilities for
five simulated error phenotypes: fusion-like, internal-disruption-like,
terminal-abnormality, cryptic-ORF-like, and repeat-like ORF. These diagnostic
classes describe sequence resemblance and are not confirmed causal annotation
errors.

The repository includes frozen binary MLP classifiers for ProtT5 (default),
ESM-2 8M, ESM-2 650M, and CARP 640M. Conditional diagnostic probabilities are
currently available for ProtT5 only.

For a small FASTA containing up to 100 proteins, open the Colab notebook using
the badge above, select a GPU runtime, upload the FASTA, and run all cells. The
notebook downloads a TSV result table and the two-stage summary plot. Use the
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
residue-weighted mean.

## Output

`embed.py` writes `embeddings.npy`, `ids.tsv`, and provenance/checksums in
`summary.json`. `score.py` writes:

- `protaudit_scores.tsv`: protein ID, binary call, protein-likeness score,
  frozen-threshold call, score band, and (for ProtT5 negatives) the predicted
  diagnostic class, its top probability, and all five class probabilities.
- `protaudit_scores_plot.png`: positive-versus-negative counts followed by the
  diagnostic-class composition of binary-negative sequences.

The frozen decision threshold is model-specific and was selected on the
validation species. Diagnostic probabilities are conditional on a sequence
receiving a binary-negative call; they should not be interpreted as biological
proof of a particular annotation error.

## Citation

If you use ProtAudit, please cite:

> Abbas, Q. et al. (2026). *ProtAudit: protein sequence auditing with protein
> language model embeddings.* Manuscript in preparation.
