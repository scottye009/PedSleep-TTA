# plotting_overlays.py
# Anonymized helper for saving and overlaying PR/ROC curves on identical axes.

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_curves(model_tag: str, label: str, y_true: np.ndarray, y_score: np.ndarray, outdir: str) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

    _ensure_dir(outdir)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    out_path = os.path.join(outdir, f"{model_tag}__{label}__curves.npz")
    np.savez_compressed(
        out_path,
        model=str(model_tag),
        label=str(label),
        pr_precision=precision.astype(np.float32),
        pr_recall=recall.astype(np.float32),
        ap=np.float32(ap),
        roc_fpr=fpr.astype(np.float32),
        roc_tpr=tpr.astype(np.float32),
        auc=np.float32(auc),
    )


def _interp_xy(x: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Monotone interpolate y(x) onto a common grid in [0,1].
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 1001, dtype=np.float32)

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    y_interp = np.interp(grid, x_sorted, y_sorted)

    return grid.astype(np.float32), y_interp.astype(np.float32)


def plot_pr_overlays(curve_files: List[str], label: str, out_png: str) -> None:
    """
    Overlay precision–recall curves from multiple models on identical axes.
    """
    plt.figure()
    for f in curve_files:
        pack = np.load(f, allow_pickle=False)
        precision = pack["pr_precision"]
        recall = pack["pr_recall"]
        ap = float(pack["ap"])
        name = str(pack["model"])

        gx, gy = _interp_xy(recall, precision)
        plt.plot(gx, gy, label=f"{name} (AP={ap:.3f})")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Overlay — {label}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc_overlays(curve_files: List[str], label: str, out_png: str) -> None:
    """
    Overlay ROC curves from multiple models on identical axes.
    """
    plt.figure()
    for f in curve_files:
        pack = np.load(f, allow_pickle=False)
        fpr = pack["roc_fpr"]
        tpr = pack["roc_tpr"]
        auc = float(pack["auc"])
        name = str(pack["model"])

        gx, gy = _interp_xy(fpr, tpr)
        plt.plot(gx, gy, label=f"{name} (AUC={auc:.3f})")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Overlay — {label}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def collect_and_plot_overlays(curves_root: str, label: str, model_tags: List[str], outdir: str) -> None:
    _ensure_dir(outdir)
    files = []
    for m in model_tags:
        f = os.path.join(curves_root, m, f"{m}__{label}__curves.npz")
        if os.path.exists(f):
            files.append(f)
    if not files:
        return

    plot_pr_overlays(files, label, os.path.join(outdir, f"{label}_PR_overlay.png"))
    plot_roc_overlays(files, label, os.path.join(outdir, f"{label}_ROC_overlay.png"))
