# 3D MedDiffusion baseline — DecoVAE 통합 노트

저널 확장 baseline (Wang et al., TMI 2025, ShanghaiTech-IMPACT).

## Setup (fresh clone)

```bash
bash scripts/setup_3d_meddiff.sh
```

위 스크립트가 자동으로:
1. `external/3d_meddiff/` 아래 upstream repo clone
2. 이 디렉토리의 `*.patch` 파일을 `patch -p1`로 적용
3. conda env 안내 출력

`external/`는 `.gitignored` — 재현은 `setup_3d_meddiff.sh`가 단일 진실이다.

## 환경 충돌

| | DecoVAE | 3D MedDiff |
|---|---|---|
| Python | 3.12 | 3.11.11 |
| torch | 2.6+ | 2.2.0 |
| CUDA | 12.4 | 12.1 |

→ **별도 conda env (`3d_meddiff`)** 필수.

```bash
conda create -n 3d_meddiff python=3.11.11 -y
conda activate 3d_meddiff
pip install -r external/3d_meddiff/requirements.txt
```

## 적용 패치 목록

- `requirements.patch` — upstream의 conda local `file://` 경로를 PyPI 핀으로 교체 + `cu121` PyPI extra-index 추가
- `vqgan_4x.patch` — `VQGANDataset_4x`가 JSON에 `{"train": [...], "val": [...]}` 명시 split 스키마를 인식 + 디렉토리/파일 경로 양쪽 허용 (upstream은 디렉토리 + 끝 40개만 val로 가정). 추가로 `tio.RescaleIntensity(percentiles=(0.5, 99.5))`로 입력 정규화 ([0,1] → 이후 `*2-1` → [-1,1]). upstream의 `*2-1`은 [0,1] 입력을 가정하지만 실제 raw NIfTI는 [0,1800]+ 범위라 recon scale이 깨졌음. (sanity v2 215834에서 검증)
- `patchvolume.patch` — `PatchVolumeAE.forward`의 val 경로에서 입력 텐서를 `patch_size`의 배수로 crop (UKB 218 등 비배수 axis 처리)

## Data bridge

DecoVAE manifest (CSV + adapter) → 3D MedDiff `data.json`:

```bash
python scripts/build_3d_meddiff_data_json.py \
    --dataset ukb_20252 \
    --train-csv ${TRAIN_CSV} \
    --valid-csv ${VALID_CSV} \
    --data-dir ${DATA_DIR} \
    --output configs/3d_meddiff/data_ukb.json
```

→ 동일 train/val split 보장 (apples-to-apples).

## 학습 launch

```bash
# default: UKB c=4, gpu-8farm, 8 GPU, 100k steps, SIGUSR1 + requeue
sbatch train_3d_meddiff.sh

# 다른 dataset
CONFIG=configs/3d_meddiff/PatchVolume_4x_ixi.yaml   sbatch train_3d_meddiff.sh
CONFIG=configs/3d_meddiff/PatchVolume_4x_brats.yaml sbatch train_3d_meddiff.sh

# Sanity (4 GPU × 100 step, gpu-4farm)
CONFIG=configs/3d_meddiff/PatchVolume_4x_ukb_sanity.yaml EXP_NAME=ukb_c4_sanity \
sbatch --partition=gpu-4farm --gres=gpu:h100:4 --ntasks-per-node=4 --cpus-per-task=14 --time=02:00:00 \
       train_3d_meddiff.sh
```

**주의**: PL DDP는 `--ntasks-per-node == cfg.model.gpus` 필요. 기본 8 GPU 외 GPU 수로 돌릴 땐 두 값을 모두 override.

## Requeue auto-resume

`train_3d_meddiff.sh`는 launcher 진입 시 `${default_root_dir}/my_model/version_*/checkpoints/*.ckpt`에서 mtime이 가장 최근인 ckpt를 찾아 임시 yaml로 `resume_from_checkpoint`를 주입한다. 따라서:

- 1-day 한계 hit → SIGUSR1 → PL checkpoint → `scontrol requeue` → 동일 job_id로 재시작 시 마지막 ckpt부터 이어짐.
- 처음 실행은 ckpt가 없으니 fresh start.
- 임시 yaml은 `${LOGS_DIR}/config_resume_${SLURM_JOB_ID}.yaml`로 보관되어 audit 가능.
