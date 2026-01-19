import { useState, useCallback, useMemo } from "react";
import { API_BASE_URL } from "../../../../lib/api-config";

/**
 * Hook for comparing metrics across multiple evaluation runs
 */
export function useMetricComparison() {
  const [comparisonRuns, setComparisonRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Add a run to comparison
  const addToComparison = useCallback(async (runId) => {
    if (comparisonRuns.some((r) => r.id === runId)) return;
    if (comparisonRuns.length >= 5) {
      setError("Maximum 5 runs can be compared at once");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`);
      if (!response.ok) throw new Error("Failed to fetch run");
      const data = await response.json();
      setComparisonRuns((prev) => [...prev, data]);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [comparisonRuns]);

  // Remove a run from comparison
  const removeFromComparison = useCallback((runId) => {
    setComparisonRuns((prev) => prev.filter((r) => r.id !== runId));
  }, []);

  // Clear all comparisons
  const clearComparison = useCallback(() => {
    setComparisonRuns([]);
    setError(null);
  }, []);

  // Compute comparison data for charts
  const comparisonData = useMemo(() => {
    if (comparisonRuns.length === 0) return null;

    const metrics = ["answer_correctness", "faithfulness", "context_precision", "answer_relevancy"];

    // Data for radar chart
    const radarData = metrics.map((metric) => {
      const point = { metric: metric.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) };
      comparisonRuns.forEach((run) => {
        point[run.name || run.id] = (run.metrics?.[metric] || 0) * 100;
      });
      return point;
    });

    // Data for bar chart comparison
    const barData = comparisonRuns.map((run) => ({
      name: run.name || run.id,
      configHash: run.config?.config_hash,
      ...Object.fromEntries(
        metrics.map((m) => [m, (run.metrics?.[m] || 0) * 100])
      ),
    }));

    // Per-question comparison
    const perQuestionData = [];
    if (comparisonRuns.length > 0) {
      const baseRun = comparisonRuns[0];
      baseRun.results?.forEach((result, idx) => {
        const questionData = {
          query: result.query,
          index: idx,
        };
        comparisonRuns.forEach((run) => {
          const runResult = run.results?.[idx];
          if (runResult) {
            questionData[`${run.name || run.id}_correctness`] =
              (runResult.scores?.answer_correctness || 0) * 100;
            questionData[`${run.name || run.id}_faithfulness`] =
              (runResult.scores?.faithfulness || 0) * 100;
          }
        });
        perQuestionData.push(questionData);
      });
    }

    // Find significant differences
    const differences = [];
    if (comparisonRuns.length >= 2) {
      metrics.forEach((metric) => {
        const values = comparisonRuns.map((r) => r.metrics?.[metric] || 0);
        const max = Math.max(...values);
        const min = Math.min(...values);
        const diff = (max - min) * 100;
        if (diff > 10) {
          differences.push({
            metric,
            difference: diff.toFixed(1),
            best: comparisonRuns[values.indexOf(max)].name || comparisonRuns[values.indexOf(max)].id,
            worst: comparisonRuns[values.indexOf(min)].name || comparisonRuns[values.indexOf(min)].id,
          });
        }
      });
    }

    return {
      radarData,
      barData,
      perQuestionData,
      differences,
      runNames: comparisonRuns.map((r) => r.name || r.id),
    };
  }, [comparisonRuns]);

  return {
    comparisonRuns,
    loading,
    error,
    addToComparison,
    removeFromComparison,
    clearComparison,
    comparisonData,
    selectedIds: comparisonRuns.map((r) => r.id),
  };
}

export default useMetricComparison;
