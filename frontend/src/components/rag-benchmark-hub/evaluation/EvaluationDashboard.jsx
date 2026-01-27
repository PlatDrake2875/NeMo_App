import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../ui/tabs";
import {
  BarChart3,
  FlaskConical,
  GitCompare,
  History,
  Plus,
  RefreshCw,
  Settings,
  TrendingUp,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";
import { StatCard } from "../../shared/StatCard";
import { useEvalRuns } from "./hooks/useEvalRuns";
import { EvalRunManager } from "./runs/EvalRunManager";
import { EvalRunComparison } from "./runs/EvalRunComparison";
import { ResultsCharts } from "./results/ResultsCharts";

/**
 * Main Evaluation Dashboard with overview stats and navigation
 */
export function EvaluationDashboard() {
  const { runs, loading, fetchRuns, selectedRunDetails, selectRun } = useEvalRuns();
  const [activeTab, setActiveTab] = useState("overview");
  const [stats, setStats] = useState({
    totalRuns: 0,
    avgPrecisionAtK: 0,
    avgRecallAtK: 0,
    recentRunsCount: 0,
  });

  // Calculate stats from runs
  useEffect(() => {
    if (runs.length === 0) {
      setStats({
        totalRuns: 0,
        avgPrecisionAtK: 0,
        avgRecallAtK: 0,
        recentRunsCount: 0,
      });
      return;
    }

    const totalRuns = runs.length;
    const avgPrecisionAtK =
      runs.reduce((sum, r) => sum + (r.metrics?.precision_at_k || 0), 0) / totalRuns;
    const avgRecallAtK =
      runs.reduce((sum, r) => sum + (r.metrics?.recall_at_k || 0), 0) / totalRuns;

    // Count runs from last 7 days
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const recentRunsCount = runs.filter(
      (r) => new Date(r.created_at) > weekAgo
    ).length;

    setStats({
      totalRuns,
      avgPrecisionAtK: avgPrecisionAtK * 100,
      avgRecallAtK: avgRecallAtK * 100,
      recentRunsCount,
    });
  }, [runs]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <FlaskConical className="h-6 w-6" />
            Evaluation Dashboard
          </h2>
          <p className="text-muted-foreground">
            Run evaluations, compare results, and track RAG pipeline performance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={fetchRuns} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatCard
          title="Total Evaluations"
          value={stats.totalRuns}
          icon={History}
          subtitle="All-time runs"
        />
        <StatCard
          title="Avg Precision@K"
          value={`${stats.avgPrecisionAtK.toFixed(1)}%`}
          icon={TrendingUp}
          valueClassName={stats.avgPrecisionAtK >= 70 ? "text-green-600" : stats.avgPrecisionAtK >= 50 ? "text-yellow-600" : "text-red-600"}
        />
        <StatCard
          title="Avg Recall@K"
          value={`${stats.avgRecallAtK.toFixed(1)}%`}
          icon={BarChart3}
          valueClassName={stats.avgRecallAtK >= 70 ? "text-green-600" : stats.avgRecallAtK >= 50 ? "text-yellow-600" : "text-red-600"}
        />
        <StatCard
          title="Recent Runs"
          value={stats.recentRunsCount}
          icon={FlaskConical}
          subtitle="Last 7 days"
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:grid-cols-none lg:inline-flex">
          <TabsTrigger value="overview" className="gap-2">
            <Settings className="h-4 w-4" />
            Run Evaluation
          </TabsTrigger>
          <TabsTrigger value="history" className="gap-2">
            <History className="h-4 w-4" />
            History
          </TabsTrigger>
          <TabsTrigger value="compare" className="gap-2">
            <GitCompare className="h-4 w-4" />
            Compare
          </TabsTrigger>
          <TabsTrigger value="charts" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            Visualizations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-6">
          {/* This would load the existing EvaluationPage content */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Evaluation</CardTitle>
              <CardDescription>
                Configure and run a new evaluation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Use the full Evaluation page from the sidebar to configure and run evaluations.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <EvalRunManager
            runs={runs}
            loading={loading}
            onRefresh={fetchRuns}
            onSelectRun={selectRun}
          />
        </TabsContent>

        <TabsContent value="compare" className="mt-6">
          <EvalRunComparison runs={runs} />
        </TabsContent>

        <TabsContent value="charts" className="mt-6">
          <ResultsCharts runs={runs} selectedRun={selectedRunDetails} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default EvaluationDashboard;
