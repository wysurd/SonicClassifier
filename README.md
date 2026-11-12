# SonicClassifier

An educational PyTorch CNN for ten-genre music classification, with consistent log-mel
preprocessing, reproducible song splits, evaluation reports, and a Streamlit explorer.
**No trained model or GTZAN performance result is provided.** Synthetic checks verify
software behavior only; they do not establish real-world genre accuracy.

## Install

Use Python 3.10–3.12 (Python 3.12 tested). On macOS Apple Silicon use a native arm64
Python and a recent macOS version supported by PyTorch MPS. CPU works on Linux and Mac.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For Linux CPU-only installations, install PyTorch first using the official CPU wheel index:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then install
requirements. Dependency ranges allow supported updates. The tested Linux CPU environment is pinned in
`requirements-linux-cpu.lock`; reproduce it with `pip install -r requirements-linux-cpu.lock
--extra-index-url https://download.pytorch.org/whl/cpu`. On Apple Silicon use `requirements.txt`
(the CPU wheel pin is Linux-specific), and capture `pip freeze` with experiment results. No API keys or external services are required.

## GTZAN: obtain and arrange locally

GTZAN is a research dataset of roughly 1,000 thirty-second recordings, 100 in each genre.
See the [official collection page](https://marsyas.info/downloads/datasets.html) for provenance
and available downloads. A commonly used mirror is
[Kaggle GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
Download manually under the provider's terms; Kaggle may require an account. Extract the
**audio**, not just feature CSVs or spectrogram images, to:

```text
data/genres_original/
  blues/blues.00000.wav
  classical/classical.00000.wav
  country/...
  disco/...
  hiphop/...
  jazz/...
  metal/...
  pop/...
  reggae/...
  rock/...
```

The original `.au` distribution is also supported, as are FLAC, MP3 and OGG. Pass the directory
whose immediate children are the ten genre folders. Downloading and redistribution rights
are your responsibility: do not commit audio, archives, datasets, or model artifacts. `data/`,
`artifacts/`, common audio extensions, and `.pt` checkpoints are ignored.

Some distributions include corrupt audio (often reported for `jazz.00054.wav`). Decoding
errors deliberately fail with the offending path instead of silently changing the dataset.
Verify the download or exclude the damaged file locally and document the exclusion. Do not
substitute synthetic data for a real benchmark result.

## Train and evaluate

```bash
python -m sonicclassifier.train --data data/genres_original --output artifacts/run --epochs 20 --seed 42
python -m sonicclassifier.evaluate --data data/genres_original --run artifacts/run
streamlit run app.py
```

Choose a new, empty output directory for every experiment. Training saves `splits.json`,
`config.json`, `history.json`, `history.png`, `best.pt`, and `last.pt`. The best validation
song accuracy selects the checkpoint; the held-out test set is evaluated only afterward.
`metrics.json` includes accuracy and per-class, macro and weighted precision, recall, F1 and
support. `confusion_matrix.png` shows true labels by row and predictions by column. The
separate evaluation command reproduces held-out evaluation with the saved split.
Checkpoints include model/optimizer states and preprocessing metadata; automatic resume is
not implemented. Load only trusted local checkpoints.

Default preprocessing: mono, 22,050 Hz, non-overlapping 3-second segments with final zero
padding, 2,048-point FFT, 512-sample hop, 64 mel bands, log power clipped to an 80 dB range
and scaled to [0, 1]. No statistics are fitted on held-out audio. The CNN uses three
3×3 convolution/ReLU/2×2 max-pool blocks (16/32/64 channels), global average pooling,
dropout, and ten output logits. Cross entropy and Adam train the network. Features are
precomputed in RAM: GTZAN requires roughly 1–2 GB plus model/runtime memory; larger collections
need a lazy or disk-cached dataset implementation.

Files are stratified into approximately 70/15/15 train/validation/test sets **before**
segmentation. Each song's segments remain together. Small datasets need enough songs in
every genre for stratification (the smoke test uses ten per genre). Predictions average
softmax probabilities across segments, so evaluation counts songs, not segments. Keep files
and their names unchanged to reproduce a saved split.

Seeds cover Python, NumPy, PyTorch, and DataLoader shuffling. Deterministic algorithms are
requested with warnings for unsupported operations. Exact results across library versions,
CPU and MPS are not guaranteed. `--device auto` uses MPS when available and otherwise CPU;
use `--device cpu` for unsupported MPS operations. MPS hardware was not tested in the Linux
development environment. No CUDA requirement is imposed.

## Explore and learn

After training, enter the checkpoint path in the Streamlit sidebar and upload an audio
clip. The app shows the first segment's mel spectrogram and mean genre probabilities over
all segments. Use short uploads (20 MB maximum); decoded duration also affects memory.
The educational panel explains convolution, learned filters, pooling, backpropagation,
and the relationship between time-frequency patterns and CNN features.

## Evaluation limitations

GTZAN has known duplicates, artist overlap and labeling problems. Keeping segments together
prevents same-file leakage, but does not remove duplicates across files or artists across
splits. For stronger claims, audit duplicates and use artist/disjoint recording groups;
report exclusions and split protocol. Genre labels are subjective and this ten-class model
cannot detect unknown genres. Softmax scores are not calibrated confidence.

## Development checks

```bash
python -m pytest -q
python scripts/synthetic_smoke.py --output /tmp/sonic-smoke
```

The smoke command creates original synthetic tones outside the repository, trains one CPU
epoch, evaluates held-out synthetic songs, and exercises checkpoint inference. Its results
are plumbing checks, **not music classification performance**. Tests cover resampling,
mono conversion, fixed feature shape, song split separation, gradient updates, checkpoint
round trips, song-level metric aggregation and Streamlit startup. No copyrighted audio is
needed. The repository has no pretrained weights; train on a legally obtained dataset
before presenting music predictions as meaningful.
