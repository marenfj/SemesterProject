from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("analysis_outputs")
SCORES_CSV = OUTPUT_ROOT / "detection_scores.csv"
METRICS_CSV = OUTPUT_ROOT / "detection_metrics.csv"

# Models output scores on different scales:
#   AudioSeal: presence probability in [0, 1] -> 0.5 is the natural cut.
#   WavMark:   bit accuracy in [0, 1] -> ~0.5 is random, so 0.75 separates noise from signal.
DEFAULT_THRESHOLDS = {"audioseal": 0.5, "wavmark": 0.75}


def load_scores() -> list[dict]:
    rows: list[dict] = []
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model": r["model"],
                "dataset": r["dataset"],
                "stem": r["stem"],
                "label": int(r["label"]),
                "score": float(r["score"]),
            })
    return rows


def filter_rows(rows: list[dict], model: str | None = None, dataset: str | None = None) -> list[dict]:
    out = []
    for r in rows:
        if model is not None and r["model"] != model:
            continue
        if dataset is not None and r["dataset"] != dataset:
            continue
        out.append(r)
    return out


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    P = float((labels == 1).sum())
    N = float((labels == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = np.concatenate(([0.0], tps / P))
    fpr = np.concatenate(([0.0], fps / N))
    return fpr, tpr


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = roc_curve(labels, scores)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def accuracy_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return float((pred == np.asarray(labels)).mean())


def metrics_for(rows: list[dict], threshold: float) -> dict:
    labels = np.array([r["label"] for r in rows], dtype=int)
    scores = np.array([r["score"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "threshold": threshold,
        "accuracy": accuracy_at_threshold(labels, scores, threshold),
        "auc": auc_score(labels, scores),
    }


def plot_roc(curves: dict, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _curve_for(rows: list[dict], model: str, dataset: str):
    sub = filter_rows(rows, model=model, dataset=dataset)
    if not sub:
        return None
    labels = np.array([r["label"] for r in sub], dtype=int)
    scores = np.array([r["score"] for r in sub], dtype=float)
    fpr, tpr = roc_curve(labels, scores)
    return fpr, tpr, auc_score(labels, scores)


def detection_metrics() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_scores()
    if not rows:
        print(f"No rows in {SCORES_CSV} — run detection first.")
        return METRICS_CSV

    summary: list[dict] = []

    # Per-model per-dataset (the four core comparison cells).
    for model in ["audioseal", "wavmark"]:
        threshold = DEFAULT_THRESHOLDS[model]
        for dataset in ["bonafide", "spoofed"]:
            sub = filter_rows(rows, model=model, dataset=dataset)
            if not sub:
                continue
            m = metrics_for(sub, threshold)
            m["group"] = f"{model}/{dataset}"
            summary.append(m)

        # Per-model combined (both datasets), for completeness.
        sub = filter_rows(rows, model=model)
        if sub:
            m = metrics_for(sub, threshold)
            m["group"] = f"{model}/all"
            summary.append(m)

    fieldnames = ["group", "n", "n_pos", "n_neg", "threshold", "accuracy", "auc"]
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary:
            writer.writerow({k: r[k] for k in fieldnames})

    # ROC plots: cross-model comparison within each dataset.
    for dataset in ["bonafide", "spoofed"]:
        curves = {}
        for model in ["audioseal", "wavmark"]:
            c = _curve_for(rows, model, dataset)
            if c is not None:
                curves[model] = c
        if curves:
            plot_roc(curves, f"AudioSeal vs WavMark — {dataset}", OUTPUT_ROOT / f"roc_models_{dataset}.png")

    # ROC plots: cross-dataset comparison within each model.
    for model in ["audioseal", "wavmark"]:
        curves = {}
        for dataset in ["bonafide", "spoofed"]:
            c = _curve_for(rows, model, dataset)
            if c is not None:
                curves[dataset] = c
        if curves:
            plot_roc(curves, f"{model} — bonafide vs spoofed", OUTPUT_ROOT / f"roc_datasets_{model}.png")

    print("\nDetection metrics:")
    print(f"{'group':<22} {'n':>5} {'n_pos':>6} {'n_neg':>6} {'thr':>5} {'acc':>8} {'AUC':>8}")
    for r in summary:
        print(
            f"{r['group']:<22} {r['n']:>5d} {r['n_pos']:>6d} {r['n_neg']:>6d} "
            f"{r['threshold']:>5.2f} {r['accuracy']:>8.4f} {r['auc']:>8.4f}"
        )
    print(f"\nWrote {METRICS_CSV}")
    return METRICS_CSV


if __name__ == "__main__":
    detection_metrics()
