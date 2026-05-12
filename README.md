# DecoVAE

3D medical-image latent generative modeling with a VAE encoder/decoder, an
optional decoder-side fine-tuning stage, and a rectified-flow diffusion UNet
over the learned latents. The reference pipeline targets UK Biobank Field 20252
T1 brain MRI but the dataset adapter layer makes it straightforward to plug in
your own data.

## Pipeline at a glance

```
   raw volumes  ──►  train_VAE.py     (stage 1: VAE in pixel space)
                          │
                          ▼
                    train_DFT.py     (optional: fine-tune decoder w/ aux losses)
                          │
                          ▼
                    extract_emb.py   (encode all subjects → latent embeddings)
                          │
                          ▼
                    train_UNET.py    (rectified-flow diffusion on latents)
                          │
                          ▼
                    compute_metric.py (FID / PSNR / SSIM / LPIPS)
```

Each stage has a `.sh` SLURM launcher (`train_VAE.sh`, `train_DFT.sh`, …) and a
Python entry point. Run the Python files directly with `torchrun` if you are
not on a SLURM cluster.

## Repository layout

```
.
├── configs/ukb_20252/        # public config templates (paths are null;
│                             #   fill in via *.local.json overrides)
├── datasets/                 # pluggable dataset-adapter interface
├── patches/                  # MONAI subclasses (Apache-2.0 derivative work)
├── scripts/                  # transforms, utilities, DDP helpers
│   ├── config_utils.py       # load_json_with_local() — drives the override
│   ├── transforms.py
│   ├── utils.py
│   └── …
├── train_VAE.py / train_DFT.py / train_UNET.py
├── compute_metric.py
├── extract_emb.py
├── env.local.example.sh      # template — copy to env.local.sh and edit
├── environment.yml           # full conda env (frozen snapshot)
└── requirements.txt          # curated pip-only dependencies
```

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate decovae
```

### Pip-only

Requires Python 3.12 and a CUDA 12.4-compatible PyTorch wheel index.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## External weights: RadImageNet feature extractor

`compute_metric.py` uses a [RadImageNet](https://github.com/BMEII-AI/RadImageNet)
ResNet-50 backbone for the FID feature extractor. Download
`RadImageNet-ResNet50_notop.pth` (a [Google Drive
mirror](https://drive.google.com/uc?export=download&id=1VOWHgOq0rm7OkE_JxlWXhMAH4CvcXUHT)
is currently available) and set the absolute path in your
`configs/ukb_20252/dataset.local.json` under `feature_extractor_path`. See the
upstream repo for the original distribution and license terms.

## Per-user configuration

The repo ships with **public** config templates that have all environment- and
account-specific values nulled out. You provide your local values via two
gitignored override files:

### 1. Shell environment — `env.local.sh`

Sourced by every `*.sh` launcher to set conda activation, TMPDIR, default
experiment paths, and W&B entity.

```bash
cp env.local.example.sh env.local.sh
# edit env.local.sh: uncomment / fill in conda activate, OUTPUT_DIR_BASE,
# DATA_DIR, TRAIN_CSV, VALID_CSV, WANDB_ENTITY, …
```

### 2. JSON configs — `*.local.json`

Each `configs/.../*.json` will deep-merge a sibling `*.local.json` on top of
itself at load time. The `.local.json` files are gitignored. Templates with
the keys you need to fill in are shipped as `*.local.example.json`:

```bash
cp configs/ukb_20252/dataset.local.example.json \
   configs/ukb_20252/dataset.local.json
# edit configs/ukb_20252/dataset.local.json — wandb_entity, data_dir,
# train/valid CSVs, feature_extractor_path

cp configs/ukb_20252/vae_train_stage1.local.example.json \
   configs/ukb_20252/vae_train_stage1.local.json
# edit custom_config.output_dir
```

Anything not overridden in the `.local.json` keeps the public default.

## SLURM launchers

The shipped launchers (`train_VAE.sh`, etc.) hard-code SLURM options for the
reference cluster (`partition=P2`, specific `--exclude` list). For other
clusters either:

- Edit the `#SBATCH` block once for your cluster, or
- Override at submission time:
  `sbatch -p mypart --exclude=none train_VAE.sh`

All experiment-level knobs (`EXP_NAME`, `OUTPUT_DIR_BASE`, `LAMBDA_*`, …) are
exposed as `${VAR:=default}` defaults so you can override them via env vars at
submission time without editing the script.

### Switching VAE stages

`train_VAE.sh` / `train_DFT.sh` / `train_UNET.sh` / `compute_metric.sh` /
`extract_emb.sh` all read a single `STAGE` variable (default: `stage1`) and use
it to derive *both* the output directory and the training config:

```
OUTPUT_DIR_BASE = ${OUTPUT_ROOT}/${STAGE}
TRAIN_CFG       = configs/ukb_20252/vae_train_${STAGE}.json   # (VAE launcher)
```

To run a stage-2 experiment, override `STAGE` at submission time — output and
config switch together:

```bash
STAGE=stage2 sbatch train_VAE.sh
```

(Add `configs/ukb_20252/vae_decft_stage2.json` first if you want to run DFT on
stage 2; the repo only ships `vae_decft_stage1.json`.)

## Running without SLURM

```bash
torchrun --nproc_per_node=4 train_VAE.py \
    --dataset_config_path configs/ukb_20252/dataset.json \
    --model_config_path   configs/ukb_20252/model_fm.json \
    --train_config_path   configs/ukb_20252/vae_train_stage1.json \
    --run_name my_first_run
```

The training scripts now read `LOCAL_RANK / RANK / WORLD_SIZE` with sane
defaults, so a single-GPU `torchrun --nproc_per_node=1` works for smoke tests.

## Dataset adapter interface

If you want to use DecoVAE on a non-UKB dataset, implement a `DatasetAdapter`
subclass (see `datasets/base.py` for the surface and
`datasets/ukb_20252.py` for a reference implementation) and register it in
`datasets/__init__.py`. The training scripts call into the adapter for:

- `extract_subject_id(image_path) -> str`
- `load_manifest(csv_path, data_dir) -> list[dict]`
- `normalize_label_df(df) -> df` and `derive_conditions(row) -> list[float]`
- `meta_value_distribution(n, seed)` (optional; used by `compute_metric.py`)

## License & attribution

This project is released under the Apache License 2.0; see [`LICENSE`](LICENSE).

The files under `patches/` (`diffusion_model_unet_maisi_v2.py`,
`rflow_scheduler_v2.py`) are derivative works of the
[MONAI](https://github.com/Project-MONAI/MONAI) project, also Apache-2.0.

## Resuming from pre-refactor checkpoints

The UNet checkpoint schema was tidied for parity with the VAE / DFT keys:

| Old key (pre-refactor) | New key      |
|------------------------|--------------|
| `unet_state_dict`      | `unet`       |
| `lr_scheduler`         | `scheduler`  |

The loaders in `train_UNET.py` and `compute_metric.py` fall back to the old
keys if the new ones are missing, so existing checkpoints keep loading
without manual migration. New checkpoints are written with the new keys.

## Determinism note

`train_*.py` sets `cudnn.benchmark = True` and `cudnn.deterministic = False`
for speed. Random seeds are fixed (different per rank), so behavior is
statistically reproducible but **not** bit-exact across runs on the same seed.
Set `torch.backends.cudnn.deterministic = True` (and `benchmark = False`) if
you need bit-exact reproducibility, at the cost of ~10–20 % speed.

## Known limitations / future work

- `scripts/utils.py` is large (~750 lines) and mixes generic utilities with
  organ-mask-specific post-processing; a split into `core_utils.py`,
  `data_utils.py`, and `organ_utils.py` is planned.
- The DDP / config-loading boilerplate is duplicated across
  `train_VAE.py` / `train_DFT.py` / `train_UNET.py`; a `scripts/train_utils.py`
  extraction is planned but deferred until a test harness exists, as the
  three `load_config` variants differ in their config sections and CLI args.
- Single-process (non-DDP) execution falls back gracefully on env-var reads,
  but `dist.init_process_group("nccl")` still expects a rendezvous backend;
  use `torchrun --nproc_per_node=1` for local smoke tests.
