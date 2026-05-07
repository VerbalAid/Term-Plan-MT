#!/usr/bin/env python3
"""Overlay training loss and validation metrics from ``trainer_state.json`` files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_trainer_state(model_dir: Path) -> Path | None:
    """Pick the richest ``trainer_state.json`` under ``model_dir`` (root or checkpoint-*)."""
    model_dir = model_dir.resolve()
    direct = model_dir / "trainer_state.json"
    if direct.is_file():
        return direct
    candidates = sorted(model_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return None

    def score(p: Path) -> tuple[int, float]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            step = int(data.get("global_step", 0))
            # Prefer checkpoint that matches best_model_checkpoint when present
            best_ckpt = data.get("best_model_checkpoint")
            if isinstance(best_ckpt, str):
                try:
                    tail = Path(best_ckpt).name
                    if tail.startswith("checkpoint-"):
                        step = max(step, int(tail.split("-")[1]))
                except (IndexError, ValueError):
                    pass
            return step, float(data.get("epoch", 0.0))
        except (OSError, json.JSONDecodeError, ValueError):
            return 0, 0.0

    return max(candidates, key=score)


def iter_train_eval_logs(model_dir: Path) -> tuple[list[dict], list[dict]]:
    """Split ``log_history`` into training-only logs (have ``loss``) and eval logs (have ``eval_*``)."""
    ts_path = resolve_trainer_state(model_dir)
    if ts_path is None:
        return [], []
    data = json.loads(ts_path.read_text(encoding="utf-8"))
    hist = data.get("log_history") or []
    train_logs: list[dict] = []
    eval_logs: list[dict] = []
    for row in hist:
        if not isinstance(row, dict):
            continue
        if "loss" in row and "eval_loss" not in row:
            train_logs.append(row)
        if any(k.startswith("eval_") for k in row):
            eval_logs.append(row)
    return train_logs, eval_logs


def _pick_eval_f1(row: dict) -> float | None:
    for k in ("eval_ner_f1", "eval_f1", "eval_micro_f1"):
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])
    return None


def plot_training_curves_overlay(
    model_dirs: list[Path],
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Overlay train loss and validation F1 for multiple training runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required for plot_training_curves.py") from e

    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_f1 = ax_loss.twinx()
    colors = plt.cm.tab10.colors

    for i, raw_dir in enumerate(model_dirs):
        model_dir = raw_dir if raw_dir.is_absolute() else ROOT / raw_dir
        label = str(model_dir)
        c = colors[i % len(colors)]
        train_logs, eval_logs = iter_train_eval_logs(model_dir)
        if not train_logs and not eval_logs:
            print(f"warning: no trainer_state.json usable under {model_dir}", file=sys.stderr)
            continue

        if train_logs:
            xs = [float(r.get("epoch", 0.0)) for r in train_logs]
            ys = [float(r["loss"]) for r in train_logs]
            ax_loss.plot(xs, ys, color=c, linestyle="-", alpha=0.85, label=f"{label} train loss")

        f1_points: list[tuple[float, float]] = []
        for r in eval_logs:
            f1 = _pick_eval_f1(r)
            if f1 is not None:
                f1_points.append((float(r.get("epoch", 0.0)), f1))
        if f1_points:
            f1_points.sort(key=lambda t: t[0])
            ax_f1.plot(
                [p[0] for p in f1_points],
                [p[1] for p in f1_points],
                color=c,
                linestyle="--",
                marker="o",
                markersize=3,
                alpha=0.9,
                label=f"{label} val F1",
            )

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss", color="black")
    ax_f1.set_ylabel("validation F1", color="gray")
    ax_loss.set_title("Training curves (overlay)")
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_f1, labels_f1 = ax_f1.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_f1, labels_loss + labels_f1, loc="upper right", fontsize=7)
    fig.tight_layout()
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay training loss and validation F1 from one or more trainer_state.json trees.",
    )
    ap.add_argument(
        "--model-dir",
        nargs="+",
        required=True,
        help="One or more training output dirs (space-separated), each containing trainer_state.json.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "training_curves_overlay.png",
        help="Output PNG path.",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    plot_training_curves_overlay([Path(p) for p in args.model_dir], args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
