"""Metrics reported in Tables 6 and 7: AUROC (macro, one-vs-rest), Macro-F1,
MRR, Precision@50, PR-AUC (macro average precision) and MCC.

Pure numpy so the evaluation path does not require scikit-learn.
"""
import numpy as np

METRIC_LABELS = {"auroc": "AUROC", "macro_f1": "Macro-F1", "mrr": "MRR",
                 "p_at_50": "P@50", "pr_auc": "PR-AUC", "mcc": "MCC"}


def _auroc_binary(y, s):
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def auroc_macro(y_true, probs):
    vals = [_auroc_binary((y_true == c).astype(int), probs[:, c])
            for c in range(probs.shape[1]) if (y_true == c).any()]
    return float(np.nanmean(vals))


def macro_f1(y_true, y_pred, n_classes):
    f1 = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


def mrr(y_true, probs):
    order = np.argsort(-probs, axis=1)
    rank = (order == y_true[:, None]).argmax(axis=1) + 1
    return float(np.mean(1.0 / rank))


def precision_at_k(y_true, probs, k=50):
    """Mean over classes of the precision among that class's top-k scores."""
    out = []
    for c in range(probs.shape[1]):
        pos = (y_true == c)
        if not pos.any():
            continue
        kk = min(k, probs.shape[0])
        top = np.argsort(-probs[:, c])[:kk]
        out.append(pos[top].mean())
    return float(np.mean(out)) if out else float("nan")


def average_precision(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def pr_auc_macro(y_true, probs):
    vals = [average_precision((y_true == c).astype(int), probs[:, c])
            for c in range(probs.shape[1]) if (y_true == c).any()]
    return float(np.mean(vals))


def mcc_multiclass(y_true, y_pred, n_classes):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)
    t = C.sum(1).astype(float)
    p = C.sum(0).astype(float)
    n = C.sum().astype(float)
    c = np.trace(C).astype(float)
    num = c * n - t @ p
    den = np.sqrt(max(n * n - p @ p, 0)) * np.sqrt(max(n * n - t @ t, 0))
    return float(num / den) if den else float("nan")


def compute_all(y_true, probs, third="mrr", n_classes=541):
    y_pred = probs.argmax(1)
    out = {"auroc": auroc_macro(y_true, probs),
           "macro_f1": macro_f1(y_true, y_pred, n_classes),
           "pr_auc": pr_auc_macro(y_true, probs),
           "mcc": mcc_multiclass(y_true, y_pred, n_classes)}
    out[third] = mrr(y_true, probs) if third == "mrr" else precision_at_k(y_true, probs, 50)
    return out
