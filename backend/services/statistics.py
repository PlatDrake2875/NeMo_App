"""
Statistical Analysis Service for RAG Evaluation.

Provides statistical methods for publication-grade ML research:
- Bootstrap confidence intervals
- Paired statistical tests (t-test, Wilcoxon)
- Effect size calculations (Cohen's d)
- Multiple comparison corrections

These methods help quantify uncertainty in evaluation results and
enable rigorous comparison between different RAG configurations.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_bootstrap: int
    std_error: float
    bootstrap_distribution: Optional[list[float]] = None  # Optional: full distribution

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "std_error": self.std_error,
        }


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two runs."""

    metric_name: str
    run_a_mean: float
    run_b_mean: float
    difference: float
    p_value: Optional[float]
    effect_size: Optional[float]  # Cohen's d
    effect_size_interpretation: Optional[str]  # "small", "medium", "large"
    is_significant: bool
    test_method: str
    run_a_std: Optional[float] = None
    run_b_std: Optional[float] = None
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "run_a_mean": self.run_a_mean,
            "run_b_mean": self.run_b_mean,
            "difference": self.difference,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "is_significant": self.is_significant,
            "test_method": self.test_method,
            "run_a_std": self.run_a_std,
            "run_b_std": self.run_b_std,
            "n_samples": self.n_samples,
        }


def bootstrap_ci(
    values: list[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: Literal["mean", "median"] = "mean",
    seed: Optional[int] = None,
    return_distribution: bool = False,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a sample.

    The bootstrap method resamples with replacement to estimate
    the sampling distribution of the statistic.

    Args:
        values: Sample values to bootstrap
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        statistic: Statistic to compute ("mean" or "median")
        seed: Random seed for reproducibility
        return_distribution: Whether to include full bootstrap distribution

    Returns:
        BootstrapResult with point estimate and confidence interval
    """
    if not values:
        return BootstrapResult(
            point_estimate=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            std_error=0.0,
        )

    values_array = np.array(values)
    n = len(values_array)

    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Compute statistic function
    if statistic == "mean":
        stat_func = np.mean
    else:
        stat_func = np.median

    # Point estimate
    point_estimate = float(stat_func(values_array))

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample_indices = rng.integers(0, n, size=n)
        resample = values_array[resample_indices]
        bootstrap_stats.append(stat_func(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = float(np.percentile(bootstrap_stats, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_stats, upper_percentile))
    std_error = float(np.std(bootstrap_stats))

    return BootstrapResult(
        point_estimate=point_estimate,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        std_error=std_error,
        bootstrap_distribution=bootstrap_stats.tolist() if return_distribution else None,
    )


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """
    Perform paired t-test for dependent samples.

    Args:
        values_a: Scores from run A
        values_b: Scores from run B (must be same length)

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(values_a) != len(values_b):
        raise ValueError("Samples must have the same length for paired test")

    n = len(values_a)
    if n < 2:
        return 0.0, 1.0

    a = np.array(values_a)
    b = np.array(values_b)

    # Differences
    d = b - a
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)

    if d_std == 0:
        return 0.0, 1.0

    # t-statistic
    t_stat = d_mean / (d_std / np.sqrt(n))

    # Two-tailed p-value using approximation
    # For small samples, this is approximate. For better accuracy, use scipy.
    df = n - 1
    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed

    return float(t_stat), float(min(p_value, 1.0))


def _t_distribution_p_value(t: float, df: int) -> float:
    """
    Approximate p-value from t-distribution.

    Uses approximation formula. For production use, consider scipy.stats.t.sf
    """
    # Simple approximation using normal for large df
    if df > 100:
        # Use normal approximation
        return _normal_p_value(t)

    # For smaller df, use a rough approximation
    # This is not as accurate as scipy but works without dependencies
    x = df / (df + t * t)
    p = 0.5 * _incomplete_beta(df / 2, 0.5, x)
    return p


def _normal_p_value(z: float) -> float:
    """Approximate p-value from normal distribution (one-tailed)."""
    # Approximation of 1 - Phi(z)
    if z < 0:
        return 1 - _normal_p_value(-z)

    # Abramowitz and Stegun approximation
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1 / (1 + b0 * z)
    phi = (1 / math.sqrt(2 * math.pi)) * math.exp(-z * z / 2)
    p = phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)

    return max(0, min(1, p))


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function approximation.

    Uses continued fraction expansion.
    """
    if x == 0:
        return 0
    if x == 1:
        return 1

    # Simple approximation for our use case
    # For production, use scipy.special.betainc
    if x < (a + 1) / (a + b + 2):
        return _beta_cf(a, b, x) * (x**a) * ((1 - x) ** b) / (a * _beta_function(a, b))
    else:
        return 1 - _incomplete_beta(b, a, 1 - x)


def _beta_cf(a: float, b: float, x: float, max_iter: int = 100, tol: float = 1e-10) -> float:
    """Continued fraction for incomplete beta."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1
    d = 1 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < tol:
            return h

    return h


def _beta_function(a: float, b: float) -> float:
    """Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def wilcoxon_signed_rank_test(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test.
    Uses normal approximation for p-value.

    Args:
        values_a: Scores from run A
        values_b: Scores from run B (must be same length)

    Returns:
        Tuple of (W_statistic, p_value)
    """
    if len(values_a) != len(values_b):
        raise ValueError("Samples must have the same length")

    n = len(values_a)
    if n < 5:
        logger.warning("Wilcoxon test with n<5 may be unreliable")
        return 0.0, 1.0

    a = np.array(values_a)
    b = np.array(values_b)

    # Differences
    d = b - a

    # Remove zeros
    nonzero_mask = d != 0
    d = d[nonzero_mask]
    n_nonzero = len(d)

    if n_nonzero < 5:
        return 0.0, 1.0

    # Rank absolute differences
    abs_d = np.abs(d)
    ranks = _rank_data(abs_d)

    # Sum of ranks for positive and negative differences
    w_plus = np.sum(ranks[d > 0])
    w_minus = np.sum(ranks[d < 0])

    # W statistic (smaller of the two)
    w = min(w_plus, w_minus)

    # Normal approximation
    mean_w = n_nonzero * (n_nonzero + 1) / 4
    std_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)

    if std_w == 0:
        return float(w), 1.0

    z = (w - mean_w) / std_w
    p_value = 2 * _normal_p_value(abs(z))  # two-tailed

    return float(w), float(min(p_value, 1.0))


def _rank_data(data: np.ndarray) -> np.ndarray:
    """Rank data with ties handled by average rank."""
    n = len(data)
    indices = np.argsort(data)
    ranks = np.empty(n)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and data[indices[j]] == data[indices[j + 1]]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indices[k]] = avg_rank
        i = j + 1

    return ranks


def cohens_d(
    values_a: list[float],
    values_b: list[float],
    paired: bool = True,
) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        values_a: Scores from group/run A
        values_b: Scores from group/run B
        paired: Whether samples are paired (affects pooled std calculation)

    Returns:
        Cohen's d effect size
    """
    if not values_a or not values_b:
        return 0.0

    a = np.array(values_a)
    b = np.array(values_b)

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    if paired:
        # For paired samples, use std of differences
        d = b - a
        std = np.std(d, ddof=1)
    else:
        # For independent samples, use pooled std
        n_a, n_b = len(a), len(b)
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        std = np.sqrt(pooled_var)

    if std == 0:
        return 0.0

    return float((mean_b - mean_a) / std)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Based on Cohen's conventions:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_runs(
    run_a_scores: list[float],
    run_b_scores: list[float],
    metric_name: str,
    alpha: float = 0.05,
    test_method: Literal["t_test", "wilcoxon", "auto"] = "auto",
) -> ComparisonResult:
    """
    Compare two evaluation runs statistically.

    Args:
        run_a_scores: Per-item scores from run A
        run_b_scores: Per-item scores from run B
        metric_name: Name of the metric being compared
        alpha: Significance level (default 0.05)
        test_method: Statistical test to use

    Returns:
        ComparisonResult with statistical analysis
    """
    if len(run_a_scores) != len(run_b_scores):
        raise ValueError("Runs must have the same number of samples for paired comparison")

    n = len(run_a_scores)
    if n < 2:
        return ComparisonResult(
            metric_name=metric_name,
            run_a_mean=run_a_scores[0] if run_a_scores else 0.0,
            run_b_mean=run_b_scores[0] if run_b_scores else 0.0,
            difference=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_interpretation="negligible",
            is_significant=False,
            test_method="none",
            n_samples=n,
        )

    # Compute basic statistics
    a = np.array(run_a_scores)
    b = np.array(run_b_scores)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    std_a = float(np.std(a, ddof=1))
    std_b = float(np.std(b, ddof=1))
    difference = mean_b - mean_a

    # Auto-select test method
    if test_method == "auto":
        # Use Wilcoxon for small samples or non-normal-looking data
        if n < 30:
            test_method = "wilcoxon"
        else:
            test_method = "t_test"

    # Perform statistical test
    if test_method == "t_test":
        _, p_value = paired_t_test(run_a_scores, run_b_scores)
        method_name = "paired_t_test"
    else:
        _, p_value = wilcoxon_signed_rank_test(run_a_scores, run_b_scores)
        method_name = "wilcoxon_signed_rank"

    # Effect size
    effect = cohens_d(run_a_scores, run_b_scores, paired=True)
    effect_interp = interpret_effect_size(effect)

    # Significance
    is_significant = p_value < alpha

    return ComparisonResult(
        metric_name=metric_name,
        run_a_mean=mean_a,
        run_b_mean=mean_b,
        difference=difference,
        p_value=p_value,
        effect_size=effect,
        effect_size_interpretation=effect_interp,
        is_significant=is_significant,
        test_method=method_name,
        run_a_std=std_a,
        run_b_std=std_b,
        n_samples=n,
    )


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating significance after correction
    """
    n = len(p_values)
    if n == 0:
        return []

    adjusted_alpha = alpha / n
    return [p < adjusted_alpha for p in p_values]


def benjamini_hochberg_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    Less conservative than Bonferroni, controls false discovery rate.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Target false discovery rate

    Returns:
        Tuple of (significance list, adjusted p-values)
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n):
        rank = i + 1
        adjusted_p[sorted_indices[i]] = sorted_p[i] * n / rank

    # Ensure monotonicity (adjusted p-values should be non-decreasing)
    for i in range(n - 2, -1, -1):
        adjusted_p[sorted_indices[i]] = min(
            adjusted_p[sorted_indices[i]],
            adjusted_p[sorted_indices[i + 1]] if i < n - 1 else 1.0,
        )

    # Cap at 1.0
    adjusted_p = np.minimum(adjusted_p, 1.0)

    # Determine significance
    significant = [p < alpha for p in adjusted_p]

    return significant, adjusted_p.tolist()


def summarize_comparison(
    comparisons: list[ComparisonResult],
    alpha: float = 0.05,
) -> str:
    """
    Generate human-readable summary of multiple comparisons.

    Args:
        comparisons: List of comparison results
        alpha: Significance level

    Returns:
        Human-readable summary string
    """
    if not comparisons:
        return "No comparisons to summarize."

    # Count significant improvements/degradations
    improvements = sum(
        1 for c in comparisons
        if c.is_significant and c.difference > 0
    )
    degradations = sum(
        1 for c in comparisons
        if c.is_significant and c.difference < 0
    )
    no_change = len(comparisons) - improvements - degradations

    lines = [
        f"Comparison Summary (alpha={alpha}):",
        f"  - Significant improvements: {improvements}",
        f"  - Significant degradations: {degradations}",
        f"  - No significant change: {no_change}",
        "",
        "Metric Details:",
    ]

    for c in comparisons:
        status = "improved" if c.difference > 0 else "degraded" if c.difference < 0 else "unchanged"
        sig = " (significant)" if c.is_significant else ""
        effect = f", {c.effect_size_interpretation} effect" if c.effect_size_interpretation else ""
        lines.append(
            f"  - {c.metric_name}: {c.run_a_mean:.3f} -> {c.run_b_mean:.3f} ({status}{sig}{effect})"
        )

    return "\n".join(lines)
