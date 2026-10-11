"""
Explanation Consistency Metrics Module.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Implements:
1. Cosine Similarity between SHAP vectors phi_baseline and phi_reweighed:
   - Handle zero vectors safely:
     * If both vectors have zero norm, similarity = 1.0 (identical attribution).
     * If one has zero norm and the other is non-zero, similarity = 0.0 (orthogonal).
     * Otherwise: dot(phi_b, phi_m) / (||phi_b||_2 * ||phi_m||_2).
2. Top-K Feature Overlap (default K=5):
   - |Top5_baseline ∩ Top5_reweighed| / K based on absolute SHAP magnitude.
3. Rank Correlation (Spearman rank correlation):
   - Evaluates attribution rank alignment between baseline and reweighed models.
   - Handles ties and constant/zero-variance vectors safely.
4. Statistical Summaries and Bootstrap Confidence Intervals:
   - Mean, median, standard deviation, and 95% bootstrap confidence intervals.
"""

from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_cosine_similarity(
    phi_baseline: np.ndarray,
    phi_reweighed: np.ndarray
) -> np.ndarray:
    """
    Calculate per-applicant cosine similarity between baseline and reweighed SHAP vectors.

    Args:
        phi_baseline: Array of shape (N, P) of baseline SHAP attributions.
        phi_reweighed: Array of shape (N, P) of reweighed SHAP attributions.

    Returns:
        1D array of length N containing cosine similarity values in [-1.0, 1.0].
    """
    phi_b = np.asarray(phi_baseline, dtype=np.float64)
    phi_m = np.asarray(phi_reweighed, dtype=np.float64)

    if phi_b.shape != phi_m.shape:
        raise ValueError(f"Shape mismatch: {phi_b.shape} vs {phi_m.shape}")

    dot_products = np.sum(phi_b * phi_m, axis=1)
    norm_b = np.linalg.norm(phi_b, axis=1)
    norm_m = np.linalg.norm(phi_m, axis=1)

    denominator = norm_b * norm_m
    n_samples = phi_b.shape[0]
    cosine_sim = np.zeros(n_samples, dtype=np.float64)

    # Handle zero vectors safely
    both_zero = (norm_b == 0.0) & (norm_m == 0.0)
    one_zero = ((norm_b == 0.0) ^ (norm_m == 0.0))
    valid = ~both_zero & ~one_zero

    cosine_sim[both_zero] = 1.0
    cosine_sim[one_zero] = 0.0
    cosine_sim[valid] = np.clip(dot_products[valid] / denominator[valid], -1.0, 1.0)

    return cosine_sim


def compute_topk_overlap(
    phi_baseline: np.ndarray,
    phi_reweighed: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Calculate per-applicant Top-K feature attribution overlap.

    Top-K is defined by the K largest features by absolute SHAP magnitude.
    Overlap = |TopK_baseline ∩ TopK_reweighed| / K.

    Args:
        phi_baseline: Array of shape (N, P).
        phi_reweighed: Array of shape (N, P).
        k: Number of top features to compare (default: 5).

    Returns:
        1D array of length N containing overlap values in [0.0, 1.0].
    """
    phi_b = np.asarray(phi_baseline, dtype=np.float64)
    phi_m = np.asarray(phi_reweighed, dtype=np.float64)

    n_samples, n_features = phi_b.shape
    if phi_m.shape != (n_samples, n_features):
        raise ValueError(f"Shape mismatch: {phi_b.shape} vs {phi_m.shape}")

    actual_k = min(k, n_features)
    overlap = np.zeros(n_samples, dtype=np.float64)

    abs_b = np.abs(phi_b)
    abs_m = np.abs(phi_m)

    for i in range(n_samples):
        # argpartition or argsort for top k features
        top_b = set(np.argpartition(abs_b[i], -actual_k)[-actual_k:])
        top_m = set(np.argpartition(abs_m[i], -actual_k)[-actual_k:])
        overlap[i] = len(top_b.intersection(top_m)) / float(actual_k)

    return overlap


def compute_rank_correlation(
    phi_baseline: np.ndarray,
    phi_reweighed: np.ndarray
) -> np.ndarray:
    """
    Calculate per-applicant Spearman rank correlation between baseline and reweighed SHAP vectors.

    Handles ties and constant/zero-variance attribution vectors safely.

    Args:
        phi_baseline: Array of shape (N, P).
        phi_reweighed: Array of shape (N, P).

    Returns:
        1D array of length N containing Spearman rank correlation values in [-1.0, 1.0].
    """
    phi_b = np.asarray(phi_baseline, dtype=np.float64)
    phi_m = np.asarray(phi_reweighed, dtype=np.float64)

    n_samples, n_features = phi_b.shape
    correlations = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        vec_b = phi_b[i]
        vec_m = phi_m[i]

        std_b = np.std(vec_b)
        std_m = np.std(vec_m)

        if std_b == 0.0 and std_m == 0.0:
            # Both are constant/zero vectors: identical attribution ranking
            correlations[i] = 1.0
        elif std_b == 0.0 or std_m == 0.0:
            # One is constant while other varies
            correlations[i] = 0.0
        else:
            res = spearmanr(vec_b, vec_m)
            corr = res.statistic if hasattr(res, "statistic") else res.correlation
            correlations[i] = 0.0 if np.isnan(corr) else np.clip(corr, -1.0, 1.0)

    return correlations


def compute_bootstrap_ci(
    data: Union[np.ndarray, pd.Series],
    statistic_fn=np.mean,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute non-parametric bootstrap confidence interval for a given 1D metric sample.
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    boot_stats = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        boot_sample = rng.choice(arr, size=n, replace=True)
        boot_stats[b] = statistic_fn(boot_sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100.0 * alpha))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))

    return round(lower, 4), round(upper, 4)


def compute_consistency_metrics_dataframe(
    dataset: str,
    model: str,
    applicant_indices: List[Any],
    phi_baseline: np.ndarray,
    phi_reweighed: np.ndarray,
    sensitive_group: Optional[pd.Series] = None,
    protected_attribute_name: str = "sex",
    k: int = 5
) -> pd.DataFrame:
    """
    Generate complete individual applicant-level consistency metrics DataFrame.
    """
    cos_sim = compute_cosine_similarity(phi_baseline, phi_reweighed)
    topk_ov = compute_topk_overlap(phi_baseline, phi_reweighed, k=k)
    rank_corr = compute_rank_correlation(phi_baseline, phi_reweighed)

    df_out = pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "applicant_idx": applicant_indices,
        "cosine_similarity": np.round(cos_sim, 4),
        "top5_overlap": np.round(topk_ov, 4),
        "rank_correlation": np.round(rank_corr, 4)
    })

    if sensitive_group is not None:
        df_out[protected_attribute_name] = sensitive_group.values

    return df_out


def summarize_metric_distribution(
    series: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Summarize a metric distribution with mean, median, std, and 95% bootstrap CI.
    """
    s = series.dropna()
    mean_val = float(s.mean())
    median_val = float(s.median())
    std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0

    mean_ci_low, mean_ci_high = compute_bootstrap_ci(
        s, statistic_fn=np.mean, n_bootstraps=1000, random_state=random_state
    )
    med_ci_low, med_ci_high = compute_bootstrap_ci(
        s, statistic_fn=np.median, n_bootstraps=1000, random_state=random_state
    )

    return {
        "mean": round(mean_val, 4),
        "mean_ci_95": [mean_ci_low, mean_ci_high],
        "median": round(median_val, 4),
        "median_ci_95": [med_ci_low, med_ci_high],
        "std": round(std_val, 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "n": len(s)
    }
