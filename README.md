# DecoVAE

3D medical-image latent generative modeling with a VAE encoder/decoder whose
**channel-wise latent decorrelation** objective (L_SID / L_VAD, toggled by the
`LAMBDA_COV / LAMBDA_COR / LAMBDA_VAR` weights) yields a better-structured latent
space for downstream diffusion, an optional decoder-side fine-tuning stage, and
a rectified-flow diffusion UNet over the learned latents.

The reference pipeline reproduces the MIUA 2026 paper (UK Biobank Field 20252,
single-modality T1 brain MRI, unconditional). The dataset-adapter layer
additionally supports a **pooled multi-cohort foundation-model** regime
(`DATASET=pooled`) with **typed metadata token-set conditioning** of the
diffusion UNet — see [Multi-cohort pooled model](#multi-cohort-pooled-foundation-model).

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
├── configs/
│   ├── ukb_20252/            # reference (MIUA) config templates (paths null)
│   ├── ixi/  brats/          # additional per-cohort configs
│   └── pooled/               # multi-cohort foundation-model config
│                             #   (cached_input, conditioning token-set, bf16)
├── datasets/                 # pluggable dataset-adapter interface
│   ├── base.py               # DatasetAdapter ABC
│   ├── ukb_20252.py / ixi.py / brats.py
│   ├── pooled.py             # pooled multi-cohort cache adapter
│   └── sampling.py           # temperature-weighted distributed sampler
├── patches/                  # MONAI subclasses (Apache-2.0 derivative work)
│   ├── diffusion_model_unet_maisi_v2.py
│   ├── rflow_scheduler_v2.py
│   └── token_set_encoder.py  # typed metadata token-set conditioning encoder
├── scripts/                  # transforms, utilities, preprocessing, DDP helpers
│   ├── config_utils.py       # load_json() — JSON config loader
│   ├── transforms.py  utils.py
│   ├── build_pooled_manifest.py / build_oasis_csv.py   # manifest builders
│   ├── preproc_pipeline.py / preprocess_cache.py        # offline preprocessing cache
│   └── …
├── train_VAE.py / train_DFT.py / train_UNET.py
├── compute_metric.py / extract_emb.py
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
is currently available) and set the absolute path in your `env.local.sh` as
`FEATURE_EXTRACTOR_PATH=...`. See the upstream repo for the original
distribution and license terms.

## Per-user configuration — `env.local.sh`

The repo ships with **public** config templates that have all environment- and
account-specific values nulled out (`data_dir`, `train_label_dir`,
`valid_label_dir`, `wandb_entity`, `feature_extractor_path`). All per-user
values live in a single gitignored shell file.

```bash
cp env.local.example.sh env.local.sh
# edit env.local.sh: uncomment / fill in conda activate, OUTPUT_ROOT,
# DATA_DIR, TRAIN_CSV, VALID_CSV, WANDB_ENTITY, FEATURE_EXTRACTOR_PATH, …
```

Each `*.sh` launcher sources `env.local.sh` and forwards the values to its
Python entry point as CLI flags (`--data_dir "${DATA_DIR}"`, etc.), so the
public JSON configs in `configs/ukb_20252/` never need to be edited.

## SLURM launchers

The shipped launchers (`train_VAE.sh`, etc.) hard-code `#SBATCH` options for the
reference cluster (GPU partitions, `--gres=gpu:h100:N`, `--account=gpu`). For
other clusters either:

- Edit the `#SBATCH` block once for your cluster, or
- Override at submission time:
  `sbatch -p mypart --gres=gpu:1 train_VAE.sh`

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
- `normalize_label_df(df) -> df` and `derive_conditions(row)` — returns a fixed
  `list[float]` (per-cohort adapters) or a **typed token dict** (the pooled
  adapter; absent keys map to `None` = token not emitted)
- `meta_value_distribution(n, seed)` (optional; used by `compute_metric.py`)

## Multi-cohort pooled foundation model

Beyond the single-cohort reference pipeline, the repo supports training one
**pooled** VAE + diffusion model across multiple brain-MRI cohorts and contrasts
(T1 / T2 / FLAIR), selected with `DATASET=pooled`:

1. **Manifest** — `scripts/build_pooled_manifest.py` harmonizes per-cohort label
   CSVs into a single `pooled_manifest_{train,valid,test}.csv` (one row per
   `subject × modality`, with typed metadata columns).
2. **Offline preprocessing cache** — `scripts/preprocess_cache.py`
   (`preprocess_cache.sh`) writes each volume as a preprocessed `.npy`
   (skull-strip → N4 → rigid-to-MNI152 → 192³ → percentile-norm) plus a typed
   conditioning-token sidecar JSON. `configs/pooled` sets `cached_input: true`
   so training loads the cache directly (no re-preprocessing).
3. **Training** — `DATASET=pooled sbatch train_VAE.sh` (then `train_DFT.sh`,
   `extract_emb.sh`, `train_UNET.sh`). The VAE is unconditional; the diffusion
   UNet is conditioned on a **typed metadata token set** (modality / sex / dx /
   age / severity) via `patches/token_set_encoder.py` — present tokens only,
   mean-pooled into the time-embedding, with classifier-free-guidance token
   dropout. Configured under `configs/pooled/model_fm.json: conditioning`
   (`enabled: false` → fully unconditional, i.e. the MIUA-reproduction path).
   Imbalance is handled by a temperature-weighted sampler
   (`datasets/sampling.py`), and training runs in bf16.
4. **Per-cell evaluation** — `CELL=<cohort>_<modality> sbatch compute_metric.sh`
   computes per-(cohort × modality) FID.

The single-cohort MIUA configs (`configs/ukb_20252`) are unchanged, so the
published result remains reproducible exactly as above.

## License & attribution

This project is released under the Apache License 2.0; see [`LICENSE`](LICENSE).

The files under `patches/` (`diffusion_model_unet_maisi_v2.py`,
`rflow_scheduler_v2.py`) are derivative works of the
[MONAI](https://github.com/Project-MONAI/MONAI) project, also Apache-2.0.

## Citation

If you use this code, please cite our MIUA 2026 paper:

> *Channel-wise Latent Space Decorrelation for 3D Brain MRI Generation.*
> Medical Image Understanding and Analysis (MIUA), 2026.

(BibTeX will be added once the proceedings entry is finalized.)

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
