import { useMemo } from "react";
import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../../ui/card";
import { ChartContainer } from "../../../shared/ChartContainer";

/**
 * Visualization charts for evaluation results
 * Note: This is a simplified version. For full charts, integrate Recharts.
 */
export function ResultsCharts({ runs, selectedRun }) {
  // Calculate metric distributions across runs
  const distributionData = useMemo(() => {
    if (runs.length === 0) return null;

    const metrics = ["context_precision", "precision_at_k", "recall_at_k"];
    const distributions = {};

    metrics.forEach((metric) => {
      const values = runs.map((r) => (r.metrics?.[metric] || 0) * 100);
      distributions[metric] = {
        min: Math.min(...values).toFixed(1),
        max: Math.max(...values).toFixed(1),
        avg: (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1),
        values,
      };
    });

    return distributions;
  }, [runs]);

  // Calculate per-question analysis for selected run
  const questionAnalysis = useMemo(() => {
    if (!selectedRun?.results) return null;

    const results = selectedRun.results;
    const lowPrecision = results.filter((r) => (r.scores?.precision_at_k || 0) < 0.5);
    const lowRecall = results.filter((r) => (r.scores?.recall_at_k || 0) < 0.5);
    const highLatency = results.filter((r) => (r.latency || 0) > 5);

    return {
      total: results.length,
      lowPrecision: lowPrecision.length,
      lowRecall: lowRecall.length,
      highLatency: highLatency.length,
      worstResults: results
        .map((r, i) => ({
          index: i,
          ...r,
          avgScore:
            ((r.scores?.context_precision || 0) +
              (r.scores?.precision_at_k || 0) +
              (r.scores?.recall_at_k || 0)) /
            3,
        }))
        .sort((a, b) => a.avgScore - b.avgScore)
        .slice(0, 5),
    };
  }, [selectedRun]);

  if (runs.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-muted-foreground">
          No evaluation runs to visualize. Run an evaluation first.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Distribution Overview */}
      <div className="grid gap-4 md:grid-cols-2">
        {distributionData &&
          Object.entries(distributionData).map(([metric, data]) => (
            <ChartContainer
              key={metric}
              title={metric.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
              description={`Distribution across ${runs.length} runs`}
            >
              <div className="space-y-3 py-2">
                {/* Simple bar visualization */}
                <div className="h-8 bg-muted rounded-md overflow-hidden relative">
                  <div
                    className="h-full bg-primary/60 transition-all"
                    style={{ width: `${data.avg}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-sm font-medium">
                    {data.avg}% avg
                  </div>
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Min: {data.min}%</span>
                  <span>Max: {data.max}%</span>
                </div>
              </div>
            </ChartContainer>
          ))}
      </div>

      {/* Selected Run Analysis */}
      {questionAnalysis && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Results Analysis</CardTitle>
              <CardDescription>
                Analysis of {selectedRun.name || "selected run"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-2xl font-bold">{questionAnalysis.total}</p>
                  <p className="text-sm text-muted-foreground">Total Questions</p>
                </div>
                <div className="text-center p-4 bg-red-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-red-600">
                    {questionAnalysis.lowPrecision}
                  </p>
                  <p className="text-sm text-muted-foreground">Low P@K (&lt;50%)</p>
                </div>
                <div className="text-center p-4 bg-yellow-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-yellow-600">
                    {questionAnalysis.lowRecall}
                  </p>
                  <p className="text-sm text-muted-foreground">Low R@K (&lt;50%)</p>
                </div>
                <div className="text-center p-4 bg-orange-500/10 rounded-lg">
                  <p className="text-2xl font-bold text-orange-600">
                    {questionAnalysis.highLatency}
                  </p>
                  <p className="text-sm text-muted-foreground">High Latency (&gt;5s)</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Worst Performing Questions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Needs Attention</CardTitle>
              <CardDescription>
                Questions with lowest average scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {questionAnalysis.worstResults.map((result, idx) => (
                  <div
                    key={idx}
                    className="p-3 border rounded-lg bg-red-500/5 border-red-500/20"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{result.query}</p>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                          Answer: {result.predicted_answer?.slice(0, 150)}...
                        </p>
                      </div>
                      <div className="text-right flex-shrink-0">
                        <p className="text-sm font-medium text-red-600">
                          {(result.avgScore * 100).toFixed(0)}% avg
                        </p>
                        <p className="text-xs text-muted-foreground">
                          P@K: {((result.scores?.precision_at_k || 0) * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Metric Trends Over Time */}
      <ChartContainer
        title="Evaluation Trends"
        description="Metrics over recent evaluation runs"
      >
        <div className="space-y-2 py-4">
          {runs.slice(0, 10).map((run, idx) => (
            <div key={run.id} className="flex items-center gap-2 text-sm">
              <span className="w-32 truncate text-muted-foreground">
                {new Date(run.created_at).toLocaleDateString()}
              </span>
              <div className="flex-1 flex gap-1">
                {["precision_at_k", "recall_at_k"].map((metric) => {
                  const value = (run.metrics?.[metric] || 0) * 100;
                  return (
                    <div
                      key={metric}
                      className="h-4 rounded"
                      style={{
                        width: `${value / 2}%`,
                        backgroundColor:
                          metric === "precision_at_k"
                            ? "hsl(var(--primary))"
                            : "hsl(var(--secondary))",
                        opacity: 0.7,
                      }}
                      title={`${metric}: ${value.toFixed(1)}%`}
                    />
                  );
                })}
              </div>
              <span className="w-16 text-right font-mono text-xs">
                {((run.metrics?.precision_at_k || 0) * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </ChartContainer>
    </div>
  );
}

ResultsCharts.propTypes = {
  runs: PropTypes.array.isRequired,
  selectedRun: PropTypes.object,
};

export default ResultsCharts;
