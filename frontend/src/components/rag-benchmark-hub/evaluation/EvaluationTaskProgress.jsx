import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Play,
  Square,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import { Button } from "../../ui/button";
import { Progress } from "../../ui/progress";
import { Badge } from "../../ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { API_BASE_URL } from "../../../lib/api-config";

const STATUS_CONFIG = {
  pending: {
    icon: Clock,
    color: "text-yellow-500",
    bgColor: "bg-yellow-100 dark:bg-yellow-900/30",
    label: "Pending",
  },
  running: {
    icon: Loader2,
    color: "text-blue-500",
    bgColor: "bg-blue-100 dark:bg-blue-900/30",
    label: "Running",
    animate: true,
  },
  completed: {
    icon: CheckCircle2,
    color: "text-green-500",
    bgColor: "bg-green-100 dark:bg-green-900/30",
    label: "Completed",
  },
  failed: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-100 dark:bg-red-900/30",
    label: "Failed",
  },
  cancelled: {
    icon: Square,
    color: "text-gray-500",
    bgColor: "bg-gray-100 dark:bg-gray-900/30",
    label: "Cancelled",
  },
};

export function EvaluationTaskProgress({
  taskId,
  onComplete,
  onViewResults,
  showCard = true,
}) {
  const [task, setTask] = useState(null);
  const [error, setError] = useState(null);
  const [polling, setPolling] = useState(true);

  const fetchTask = useCallback(async () => {
    if (!taskId) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/tasks/${taskId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch task: ${response.status}`);
      }
      const data = await response.json();
      setTask(data);
      setError(null);

      // Stop polling if task is complete
      if (["completed", "failed", "cancelled"].includes(data.status)) {
        setPolling(false);
        if (data.status === "completed" && onComplete) {
          onComplete(data);
        }
      }
    } catch (err) {
      setError(err.message);
    }
  }, [taskId, onComplete]);

  // Poll for updates
  useEffect(() => {
    if (!taskId || !polling) return;

    fetchTask();
    const interval = setInterval(fetchTask, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [taskId, polling, fetchTask]);

  const handleCancel = async () => {
    if (!taskId) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/evaluation/tasks/${taskId}/cancel`,
        { method: "POST" }
      );
      if (response.ok) {
        fetchTask();
      }
    } catch (err) {
      console.error("Failed to cancel task:", err);
    }
  };

  const handleViewResults = () => {
    if (task?.result_run_id && onViewResults) {
      onViewResults(task.result_run_id);
    }
  };

  if (!taskId) {
    return null;
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg border border-red-200 bg-red-50 dark:bg-red-900/20">
        <p className="text-red-600 dark:text-red-400">Error: {error}</p>
        <Button variant="outline" size="sm" onClick={fetchTask} className="mt-2">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-muted-foreground">Loading task...</span>
      </div>
    );
  }

  const statusConfig = STATUS_CONFIG[task.status] || STATUS_CONFIG.pending;
  const StatusIcon = statusConfig.icon;

  const content = (
    <div className="space-y-4">
      {/* Status and Progress */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <StatusIcon
            className={`h-5 w-5 ${statusConfig.color} ${
              statusConfig.animate ? "animate-spin" : ""
            }`}
          />
          <span className="font-medium">{statusConfig.label}</span>
          {task.status === "running" && (
            <Badge variant="secondary" className="ml-2">
              {task.current_pair} / {task.total_pairs}
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-2">
          {task.status === "running" && (
            <Button variant="outline" size="sm" onClick={handleCancel}>
              <Square className="h-4 w-4 mr-1" />
              Cancel
            </Button>
          )}
          {task.status === "completed" && task.result_run_id && (
            <Button variant="default" size="sm" onClick={handleViewResults}>
              <ExternalLink className="h-4 w-4 mr-1" />
              View Results
            </Button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {(task.status === "running" || task.status === "pending") && (
        <div className="space-y-2">
          <Progress value={task.progress_percent} className="h-2" />
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>{task.current_step || "Initializing..."}</span>
            <span>{task.progress_percent.toFixed(1)}%</span>
          </div>
        </div>
      )}

      {/* Task Info */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">Dataset:</span>
          <span className="ml-2 font-medium">{task.eval_dataset_name || "N/A"}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Collection:</span>
          <span className="ml-2 font-medium">
            {task.collection_display_name?.split("_").pop() || "N/A"}
          </span>
        </div>
        {task.started_at && (
          <div>
            <span className="text-muted-foreground">Started:</span>
            <span className="ml-2">
              {new Date(task.started_at).toLocaleTimeString()}
            </span>
          </div>
        )}
        {task.completed_at && (
          <div>
            <span className="text-muted-foreground">Completed:</span>
            <span className="ml-2">
              {new Date(task.completed_at).toLocaleTimeString()}
            </span>
          </div>
        )}
      </div>

      {/* Error Message */}
      {task.error_message && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200">
          <p className="text-sm text-red-600 dark:text-red-400">
            {task.error_message}
          </p>
        </div>
      )}
    </div>
  );

  if (!showCard) {
    return content;
  }

  return (
    <Card className={statusConfig.bgColor}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Play className="h-5 w-5" />
          Evaluation Task
        </CardTitle>
        <CardDescription>Task ID: {task.id.slice(0, 8)}...</CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}

EvaluationTaskProgress.propTypes = {
  taskId: PropTypes.string,
  onComplete: PropTypes.func,
  onViewResults: PropTypes.func,
  showCard: PropTypes.bool,
};

export default EvaluationTaskProgress;
