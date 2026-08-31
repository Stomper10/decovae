# Archived launchers

Superseded run scripts, kept because each one is the **provenance of a decision that is
still load-bearing**. Nothing here is dead code to be deleted on sight — if you need to
answer "where did this number come from", the answer is often in this directory.

To reuse one, move it back to `scripts/`; the config paths inside are repo-root-relative
(every launcher `cd`s to `SLURM_SUBMIT_DIR`), so they still resolve from here unchanged.

| script | what it established | closed by |
|---|---|---|
| `train_VAE_klprobe.sh` | first kl band sweep (1e-3 … 4e-3) at eff32 | kl 5e-3 found over-regularizing at 170k |
| `train_VAE_klprobe2.sh` | second, finer sweep (5e-4 … 9e-4); gstd is U-shaped in kl and the band is only 8e-4–1e-3 | **kl = 8e-4 adopted** as the band kl for all three regularizer families |
| `train_VAE_packprobe.sh` | 3-group co-residency works on one node (~11 GB/GPU) | the recipe now inlined in `train_VAE_pack.sh` |
| `train_VAE_pairprobe.sh` | 2-into-1 GPU *splitting* halves the effective batch | pair pattern restricted to stage1 ablations, banned for UNet/stage2 |
| `train_VAE_scaleprobe.sh` | co-residency scales to 5 groups, dies at 6; the limit is contention, not memory | sweeps now run in waves of ≤5 |
| `train_VAE_split.sh` | separate-validation-split probe | folded into the standard manifest |
| `train_VAE_4cell.sh` | 4-cell subset run, dilution check for the rFID floor | floor was uniform 14→4 cells; dilution ruled out |
| `train_VAE_pack2.sh` | VAD λ bracket (5/5 and 10/10) at kl 8e-4 | ran to 320k; evaluated 2026-08-31 — revealed the VAD inversion |
| `train_VAE_p96pack.sh` | 96³ vs 64³ patch lever | `journal_plan/results_p96.xlsx` |
| `train_VAE_datamix_pack.sh` | ukbT1 / +adniT1 / +ukbFLAIR specialist-vs-pooled | `journal_plan/results_datamix_percell_s1.csv` |
| `train_VAE_kl5e4pack.sh` | kl 5e-4 × {p64,p96} × {ukbT1,pooled} 2×2 | paused at 320k/150k/110k/110k; superseded once kl 8e-4 won the band. Checkpoints remain on disk, so it can resume state-based if the question ever returns. |
| `eval_interim_trajectory.sh` | per-arm interim-checkpoint eval driver | superseded by `scripts/eval_triad_trajectory.sh` (multi-arm, SSIM_FG, 4 extractors) |
| `stage_interim_ckpts.sh` | staged interim ckpts to GSDS for the above | same |
| `fid_decompose_cached.py` | FID decomposition (mean-shift vs covariance term) on cached activations | `journal_plan/fid_cause_diag_report.md` |

## Still live in `scripts/`

`train_VAE_pack.sh` (canonical stage1 3-pack, and the template for any future
regularizer arm) · `train_UNET_pack.sh` (diffusion) · `train_DFT_pooled.sh` (decoder
fine-tuning) · `eval_triad_trajectory.sh` (canonical eval driver) ·
`eval_recon_eyeball.sh` (visual spot-check).

The configs these archived launchers reference are deliberately **left in
`configs/pooled/`** rather than moved alongside them — they are small, they sort together
by name (`vae_train_stage1_eff32_kl*.json`), and moving them would break the paths inside
the archived scripts.
