import { useState, useEffect, useCallback, useRef } from "react";
import PropTypes from "prop-types";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Square,
  RefreshCw,
  Play,
  Eye,
  Trash2,
  MoreHorizontal,
} from "lucide-react";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import { Progress } from "../../ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../../ui/dropdown-menu";
import { getApiBaseUrl } from "../../../lib/api-config";

const STATUS_CONFIG = {
  pending: {
    icon: Clock,
    color: "text-yellow-500",
    bgColor: "bg-yellow-50 dark:bg-yellow-950/20",
    borderColor: "border-yellow-200",
    label: "Pending",
    badgeVariant: "secondary",
  },
  running: {
    icon: Loader2,
    color: "text-blue-500",
    bgColor: "bg-blue-50 dark:bg-blue-950/20",
    borderColor: "border-blue-200",
    label: "Running",
    badgeVariant: "default",
    animate: true,
  },
  completed: {
    icon: CheckCircle2,
    color: "text-green-500",
    bgColor: "bg-green-50 dark:bg-green-950/20",
    borderColor: "border-green-200",
    label: "Completed",
    badgeVariant: "success",
  },
  failed: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-50 dark:bg-red-950/20",
    borderColor: "border-red-200",
    label: "Failed",
    badgeVariant: "destructive",
  },
  cancelled: {
    icon: Square,
    color: "text-gray-500",
    bgColor: "bg-gray-50 dark:bg-gray-950/20",
    borderColor: "border-gray-200",
    label: "Cancelled",
    badgeVariant: "outline",
  },
};

// Mini metric bar component
function MetricBar({ label, value, color }) {
  const percentage = (value * 100).toFixed(0);
  const barColor = value >= 0.7 ? "bg-green-500" : value >= 0.4 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div className="flex-1 min-w-[80px]">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} rounded-full transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function JobCard({ task, onViewResults, onCancel, onRetry }) {
  const statusConfig = STATUS_CONFIG[task.status] || STATUS_CONFIG.pending;
  const StatusIcon = statusConfig.icon;

  const formatTime = (isoString) => {
    if (!isoString) return "";
    const date = new Date(isoString);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getDuration = () => {
    if (!task.started_at) return null;
    const start = new Date(task.started_at);
    const end = task.completed_at ? new Date(task.completed_at) : new Date();
    const seconds = Math.floor((end - start) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const metrics = task.metrics;
  const isCompleted = task.status === "completed";
  const isRunning = task.status === "running" || task.status === "pending";
  const isFailed = task.status === "failed" || task.status === "cancelled";

  return (
    <Card className={`group hover:shadow-md transition-all duration-200 ${
      isCompleted ? "bg-card border-l-4 border-l-green-500" :
      isRunning ? "bg-blue-50/50 dark:bg-blue-950/10 border-l-4 border-l-blue-500" :
      isFailed ? "bg-red-50/50 dark:bg-red-950/10 border-l-4 border-l-red-500" :
      "bg-card"
    }`}>
      <CardContent className="p-4">
        {/* Header Row */}
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <div className={`${statusConfig.color} flex-shrink-0`}>
              <StatusIcon className={`h-5 w-5 ${statusConfig.animate ? "animate-spin" : ""}`} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-semibold text-sm truncate">
                  {task.eval_dataset_name || "Quick Test"}
                </h3>
                {!isCompleted && (
                  <Badge variant={statusConfig.badgeVariant} className="text-xs">
                    {statusConfig.label}
                  </Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground truncate">
                {task.collection_display_name?.split("_").slice(-2).join("_") || "N/A"}
                {task.pair_count && <span className="ml-2">• {task.pair_count} pairs</span>}
                {task.config?.llm_model && (
                  <span className="ml-2 font-mono">• {task.config.llm_model.split("/").pop()}</span>
                )}
              </p>
              {(task.result_run_id || task.run_id) && (
                <p className="text-xs text-muted-foreground font-mono mt-0.5">
                  Run ID: <span className="text-foreground select-all">{task.result_run_id || task.run_id}</span>
                </p>
              )}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-1 flex-shrink-0">
            {isCompleted && task.result_run_id && (
              <Button
                variant="default"
                size="sm"
                onClick={() => onViewResults?.(task.result_run_id)}
                className="shadow-sm"
              >
                <Eye className="h-4 w-4 mr-1" />
                View
              </Button>
            )}
            {isRunning && (
              <Button variant="outline" size="sm" onClick={() => onCancel?.(task.id)}>
                <Square className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            )}
            {isFailed && (
              <Button variant="outline" size="sm" onClick={() => onRetry?.(task)}>
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry
              </Button>
            )}
          </div>
        </div>

        {/* Progress for running tasks */}
        {isRunning && (
          <div className="mb-3 p-3 bg-blue-100/50 dark:bg-blue-900/20 rounded-lg">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-muted-foreground">{task.current_step || "Processing..."}</span>
              <span className="font-medium">{task.current_pair || 0}/{task.total_pairs || 0} ({task.progress_percent?.toFixed(0) || 0}%)</span>
            </div>
            <Progress value={task.progress_percent || 0} className="h-2" />
          </div>
        )}

        {/* Error message for failed tasks */}
        {task.error_message && (
          <div className="mb-3 p-2 bg-red-100 dark:bg-red-900/20 rounded text-xs text-red-700 dark:text-red-400">
            <div className="font-mono text-muted-foreground mb-1">Task ID: {task.id}</div>
            {task.error_message}
          </div>
        )}

        {/* Metrics for completed runs */}
        {isCompleted && metrics && (
          <div className="mb-3 p-3 bg-muted/50 rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {metrics.answer_correctness != null && (
                <MetricBar label="Correctness" value={metrics.answer_correctness} />
              )}
              {metrics.faithfulness != null && (
                <MetricBar label="Faithfulness" value={metrics.faithfulness} />
              )}
              {metrics.answer_relevancy != null && (
                <MetricBar label="Relevancy" value={metrics.answer_relevancy} />
              )}
              {metrics.context_precision != null && (
                <MetricBar label="Ctx Precision" value={metrics.context_precision} />
              )}
            </div>
          </div>
        )}

        {/* Config tags and metadata */}
        <div className="flex items-center justify-between gap-4 flex-wrap">
          {/* Hyperparameters */}
          {task.config && (
            <div className="flex items-center gap-1.5 flex-wrap">
              {task.config.use_rag !== undefined && (
                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                  task.config.use_rag
                    ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                    : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
                }`}>
                  RAG {task.config.use_rag ? "ON" : "OFF"}
                </span>
              )}
              {task.config.use_colbert && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">
                  ColBERT
                </span>
              )}
              {task.config.top_k && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                  K={task.config.top_k}
                </span>
              )}
              {task.config.temperature !== undefined && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                  T={task.config.temperature}
                </span>
              )}
              {task.config.embedder && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 font-mono">
                  {task.config.embedder.split("/").pop()}
                </span>
              )}
            </div>
          )}

          {/* Timestamp and duration */}
          <div className="flex items-center gap-3 text-xs text-muted-foreground ml-auto">
            <span>{formatTime(task.created_at)}</span>
            {getDuration() && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {getDuration()}
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

JobCard.propTypes = {
  task: PropTypes.object.isRequired,
  onViewResults: PropTypes.func,
  onCancel: PropTypes.func,
  onRetry: PropTypes.func,
};

export function EvaluationJobsPanel({ onViewResults, onNewEvaluation, refreshTrigger }) {
  const [tasks, setTasks] = useState([]);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("all");
  const tasksRef = useRef(tasks);

  const fetchTasks = useCallback(async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/evaluation/tasks?limit=50`);
      if (response.ok) {
        const data = await response.json();
        setTasks(data);
      }
    } catch (err) {
      console.error("Failed to fetch tasks:", err);
    }
  }, []);

  const fetchRuns = useCallback(async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/evaluation/runs`);
      if (response.ok) {
        const data = await response.json();
        setRuns(data);
      }
    } catch (err) {
      console.error("Failed to fetch runs:", err);
    }
  }, []);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    await Promise.all([fetchTasks(), fetchRuns()]);
    setLoading(false);
  }, [fetchTasks, fetchRuns]);

  // Keep ref in sync with tasks state
  useEffect(() => {
    tasksRef.current = tasks;
  }, [tasks]);

  // Initial fetch and auto-refresh for running tasks
  useEffect(() => {
    fetchAll();

    const interval = setInterval(() => {
      const hasActive = tasksRef.current.some(t => t.status === "running" || t.status === "pending");
      if (hasActive) {
        fetchTasks();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchAll, fetchTasks]);

  // Refresh when triggered by parent (e.g., after starting new evaluation)
  useEffect(() => {
    if (refreshTrigger > 0) {
      fetchAll();
    }
  }, [refreshTrigger, fetchAll]);

  const handleCancel = async (taskId) => {
    try {
      await fetch(`${getApiBaseUrl()}/api/evaluation/tasks/${taskId}/cancel`, {
        method: "POST",
      });
      fetchTasks();
    } catch (err) {
      console.error("Failed to cancel task:", err);
    }
  };

  const handleRetry = (task) => {
    // Could implement retry by opening the modal with pre-filled config
    onNewEvaluation?.();
  };

  // Create a map of run_id -> run for quick lookup
  const runsById = new Map(runs.map(run => [run.id, run]));

  // Merge metrics and config from runs into tasks that have result_run_id
  const tasksWithMetrics = tasks.map(task => {
    if (task.result_run_id && runsById.has(task.result_run_id)) {
      const run = runsById.get(task.result_run_id);
      return {
        ...task,
        metrics: task.metrics || run.metrics,
        config: task.config || run.config,
        pair_count: task.pair_count || run.pair_count,
      };
    }
    return task;
  });

  // Convert runs to task-like format for unified display
  const runsAsItems = runs.map(run => ({
    id: `run-${run.id}`,
    run_id: run.id,
    status: "completed",
    eval_dataset_name: run.name,
    collection_display_name: run.config?.collection || "N/A",
    created_at: run.created_at,
    completed_at: run.created_at,
    result_run_id: run.id,
    pair_count: run.pair_count,
    metrics: run.metrics,
    config: run.config,
    isRun: true, // Flag to identify this is from runs API
  }));

  // Get task result_run_ids to avoid duplicates
  const taskRunIds = new Set(tasksWithMetrics.filter(t => t.result_run_id).map(t => t.result_run_id));

  // Filter out runs that already have a corresponding task
  const uniqueRuns = runsAsItems.filter(r => !taskRunIds.has(r.run_id));

  // Combine tasks with unique runs for the "all" and "completed" views
  const allItems = [...tasksWithMetrics, ...uniqueRuns].sort((a, b) =>
    new Date(b.created_at) - new Date(a.created_at)
  );

  // Filter items by tab
  const filteredItems = allItems.filter(item => {
    switch (activeTab) {
      case "running":
        return item.status === "running" || item.status === "pending";
      case "completed":
        return item.status === "completed";
      case "failed":
        return item.status === "failed" || item.status === "cancelled";
      default:
        return true;
    }
  });

  const runningCount = tasks.filter(t => t.status === "running" || t.status === "pending").length;
  const completedCount = allItems.filter(t => t.status === "completed").length;
  const failedCount = tasks.filter(t => t.status === "failed" || t.status === "cancelled").length;

  return (
    <div className="space-y-4">
      {/* Header with New Evaluation button */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Evaluation Jobs</h2>
          <p className="text-sm text-muted-foreground">
            {allItems.length} total jobs, {runningCount} running
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={fetchAll} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button onClick={onNewEvaluation}>
            <Play className="h-4 w-4 mr-2" />
            New Evaluation
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="all" className="relative">
            All
            <Badge variant="secondary" className="ml-2 text-xs">
              {allItems.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="running" className="relative">
            Running
            {runningCount > 0 && (
              <Badge variant="default" className="ml-2 text-xs animate-pulse">
                {runningCount}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="completed">
            Completed
            <Badge variant="secondary" className="ml-2 text-xs">
              {completedCount}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="failed">
            Failed
            {failedCount > 0 && (
              <Badge variant="destructive" className="ml-2 text-xs">
                {failedCount}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-4">
          {loading && allItems.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading jobs...</span>
            </div>
          ) : filteredItems.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <div className="text-muted-foreground">
                  {activeTab === "all" ? (
                    <>
                      <Play className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No evaluation jobs yet</p>
                      <p className="text-sm">Start a new evaluation to see it here</p>
                      <Button className="mt-4" onClick={onNewEvaluation}>
                        <Play className="h-4 w-4 mr-2" />
                        New Evaluation
                      </Button>
                    </>
                  ) : activeTab === "running" ? (
                    <>
                      <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No running jobs</p>
                    </>
                  ) : activeTab === "completed" ? (
                    <>
                      <CheckCircle2 className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No completed jobs</p>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>No failed jobs</p>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredItems.map((item) => (
                <JobCard
                  key={item.id}
                  task={item}
                  onViewResults={onViewResults}
                  onCancel={handleCancel}
                  onRetry={handleRetry}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

EvaluationJobsPanel.propTypes = {
  onViewResults: PropTypes.func,
  onNewEvaluation: PropTypes.func,
  refreshTrigger: PropTypes.number,
};

export default EvaluationJobsPanel;
