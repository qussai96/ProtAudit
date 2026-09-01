# ProtAudit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qussai96/ProtAudit/blob/main/ProtAudit_Colab.ipynb)

ProtAudit assigns each protein a score from 0 to 1: higher values indicate a
more protein-like sequence. The repository includes frozen MLP classifiers for
ProtT5 (default), ESM-2 8M, ESM-2 650M, and CARP 640M.

For a small FASTA containing up to 100 proteins, open the Colab notebook using
the badge above, select a GPU runtime, upload the FASTA, and run all cells. The
notebook downloads a TSV score table and the ProtAudit score-band plot. Use the
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

- `protaudit_scores.tsv`: protein ID, score, frozen-threshold call, and score band.
- `protaudit_scores_plot.png`: counts and percentages in the `>=0.9`, `0.5–0.9`,
  and `<0.5` score bands.

The frozen decision threshold is model-specific and was selected on the
validation species. The three plot bands are descriptive confidence bands.

## Citation

If you use ProtAudit, please cite:

> Abbas, Q. et al. (2026). *ProtAudit: protein sequence auditing with protein
> language model embeddings.* Manuscript in preparation.
