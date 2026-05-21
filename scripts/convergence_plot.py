"""Convergence diagnostic for 3D MedDiff Phase 1 (PatchVolume) training.

Parses ``val/recon_loss=...`` entries from a 3D MedDiff training log
(written by ``train_PatchVolume.py``), plots the trajectory, and reports
the slope over the last N validation samples — used to defend the
compute-matched-at-300k methodology choice (see ``journal_plan/handoff.md``
§2.8 and ``configs/3d_meddiff/PatchVolume_4x_ukb.yaml``).

Usage:
    python scripts/convergence_plot.py \\
        --log /data/wonyoungjang/decodata/3d_meddiff/ukb_c4/logs/3d_meddiff_215982.log \\
        --out journal_plan/figures/ukb_c4_convergence.png

Decision rule (from journal_methodology_decisions §2):
    If slope over last 10 unique val samples shows > 0.5% relative improvement,
    extend max_steps from 300k to 400k (hard cap). Otherwise treat as converged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VAL_RECON_RE = re.compile(rb"val/recon_loss=([0-9.eE+-]+)")


def parse_val_recon(log_path: Path) -> list[float]:
    """Read the log and return unique val/recon_loss values in order."""
    seen: set[str] = set()
    out: list[float] = []
    with log_path.open("rb") as f:
        for line in f:
            for match in VAL_RECON_RE.finditer(line):
                token = match.group(1).decode()
                if token in seen:
                    continue
                seen.add(token)
                try:
                    out.append(float(token))
                except ValueError:
                    continue
    return out


def relative_slope(values: list[float], window: int) -> float:
    """Relative improvement over the last `window` values.

    Returns (first - last) / first, so positive = still improving.
    """
    if len(values) < window:
        window = len(values)
    if window < 2:
        return 0.0
    tail = values[-window:]
    return (tail[0] - tail[-1]) / max(abs(tail[0]), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path, help="Training log path")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path (optional)")
    ap.add_argument("--window", type=int, default=10, help="Slope window (default 10 unique val samples)")
    ap.add_argument("--threshold", type=float, default=0.005, help="Relative improvement threshold for 'still improving' (default 0.005 = 0.5%%)")
    args = ap.parse_args()

    values = parse_val_recon(args.log)
    if not values:
        raise SystemExit(f"No val/recon_loss entries found in {args.log}")

    slope = relative_slope(values, args.window)
    verdict = "STILL IMPROVING — extend to 400k" if slope > args.threshold else "CONVERGED — keep at 300k"

    print(f"log         : {args.log}")
    print(f"n unique val: {len(values)}")
    print(f"first  val  : {values[0]:.4f}")
    print(f"last   val  : {values[-1]:.4f}")
    print(f"min    val  : {min(values):.4f}")
    print(f"last {args.window} mean : {sum(values[-args.window:]) / min(args.window, len(values)):.4f}")
    print(f"slope (rel) : {slope:+.4f}  (threshold {args.threshold:+.4f})")
    print(f"verdict     : {verdict}")

    if args.out is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping plot")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(values) + 1), values, lw=1.0)
    ax.axhline(min(values), color="gray", linestyle=":", lw=0.8, label=f"min = {min(values):.4f}")
    if len(values) >= args.window:
        ax.axvspan(len(values) - args.window + 1, len(values), color="orange", alpha=0.15, label=f"slope window (last {args.window})")
    ax.set_xlabel("validation sample index")
    ax.set_ylabel("val/recon_loss")
    ax.set_title(f"3D MedDiff PatchVolume convergence (slope={slope:+.4f}, {verdict.split(' — ')[0]})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"plot saved  : {args.out}")


if __name__ == "__main__":
    main()
