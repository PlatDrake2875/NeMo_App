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
import { API_BASE_URL } from "../../../lib/api-config";

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

function JobCard({ task, onViewResults, onCancel, onRetry }) {
  const statusConfig = STATUS_CONFIG[task.status] || STATUS_CONFIG.pending;
  const StatusIcon = statusConfig.icon;

  const formatTime = (isoString) => {
    if (!isoString) return "";
    const date = new Date(isoString);
    return date.toLocaleString();
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

  // Get metrics summary for completed runs
  const getMetricsSummary = () => {
    if (!task.metrics) return null;
    const correctness = task.metrics.answer_correctness;
    const faithfulness = task.metrics.faithfulness;
    if (correctness == null && faithfulness == null) return null;
    return { correctness, faithfulness };
  };

  const metrics = getMetricsSummary();

  return (
    <Card className={`${statusConfig.bgColor} ${statusConfig.borderColor} border`}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Left side - Status and Info */}
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className={`mt-0.5 ${statusConfig.color}`}>
              <StatusIcon
                className={`h-5 w-5 ${statusConfig.animate ? "animate-spin" : ""}`}
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium text-sm truncate">
                  {task.eval_dataset_name || "Quick Test"}
                </span>
                <Badge variant={statusConfig.badgeVariant} className="text-xs">
                  {statusConfig.label}
                </Badge>
                {task.pair_count && (
                  <Badge variant="outline" className="text-xs">
                    {task.pair_count} pairs
                  </Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-1 truncate">
                Collection: {task.collection_display_name?.split("_").slice(-2).join("_") || "N/A"}
              </p>
              {task.status === "running" && task.current_step && (
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {task.current_step}
                </p>
              )}
              {task.error_message && (
                <p className="text-xs text-red-600 mt-1 truncate">
                  Error: {task.error_message}
                </p>
              )}
              {/* Metrics for completed runs */}
              {metrics && (
                <div className="flex items-center gap-3 mt-1 text-xs">
                  {metrics.correctness != null && (
                    <span className={metrics.correctness >= 0.7 ? "text-green-600" : metrics.correctness >= 0.4 ? "text-yellow-600" : "text-red-600"}>
                      Correctness: {(metrics.correctness * 100).toFixed(0)}%
                    </span>
                  )}
                  {metrics.faithfulness != null && (
                    <span className={metrics.faithfulness >= 0.7 ? "text-green-600" : metrics.faithfulness >= 0.4 ? "text-yellow-600" : "text-red-600"}>
                      Faithful: {(metrics.faithfulness * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              )}
              <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                <span>{formatTime(task.created_at)}</span>
                {getDuration() && <span>Duration: {getDuration()}</span>}
              </div>
            </div>
          </div>

          {/* Right side - Progress and Actions */}
          <div className="flex flex-col items-end gap-2">
            {/* Progress for running tasks */}
            {(task.status === "running" || task.status === "pending") && (
              <div className="w-32">
                <div className="flex justify-between text-xs mb-1">
                  <span>{task.current_pair}/{task.total_pairs}</span>
                  <span>{task.progress_percent?.toFixed(0)}%</span>
                </div>
                <Progress value={task.progress_percent || 0} className="h-2" />
              </div>
            )}

            {/* Action buttons */}
            <div className="flex items-center gap-1">
              {task.status === "completed" && task.result_run_id && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onViewResults?.(task.result_run_id)}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  Results
                </Button>
              )}
              {task.status === "running" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onCancel?.(task.id)}
                >
                  <Square className="h-4 w-4 mr-1" />
                  Cancel
                </Button>
              )}
              {(task.status === "failed" || task.status === "cancelled") && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onRetry?.(task)}
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Retry
                </Button>
              )}
            </div>
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

export function EvaluationJobsPanel({ onViewResults, onNewEvaluation }) {
  const [tasks, setTasks] = useState([]);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("all");
  const tasksRef = useRef(tasks);

  const fetchTasks = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/tasks?limit=50`);
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
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs`);
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

  const handleCancel = async (taskId) => {
    try {
      await fetch(`${API_BASE_URL}/api/evaluation/tasks/${taskId}/cancel`, {
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
  const taskRunIds = new Set(tasks.filter(t => t.result_run_id).map(t => t.result_run_id));

  // Filter out runs that already have a corresponding task
  const uniqueRuns = runsAsItems.filter(r => !taskRunIds.has(r.run_id));

  // Combine tasks with unique runs for the "all" and "completed" views
  const allItems = [...tasks, ...uniqueRuns].sort((a, b) =>
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
};

export default EvaluationJobsPanel;
