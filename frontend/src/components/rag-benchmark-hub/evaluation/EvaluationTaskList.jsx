import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Square,
  RefreshCw,
  ChevronRight,
  ListTodo,
} from "lucide-react";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { API_BASE_URL } from "../../../lib/api-config";

const STATUS_ICONS = {
  pending: { icon: Clock, color: "text-yellow-500" },
  running: { icon: Loader2, color: "text-blue-500", animate: true },
  completed: { icon: CheckCircle2, color: "text-green-500" },
  failed: { icon: XCircle, color: "text-red-500" },
  cancelled: { icon: Square, color: "text-gray-500" },
};

export function EvaluationTaskList({ onSelectTask, onViewResults, limit = 10 }) {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${API_BASE_URL}/api/evaluation/tasks?limit=${limit}`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch tasks: ${response.status}`);
      }
      const data = await response.json();
      setTasks(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();

    // Auto-refresh if there are running tasks
    const interval = setInterval(() => {
      const hasRunning = tasks.some((t) => t.status === "running");
      if (hasRunning) {
        fetchTasks();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Re-fetch when tasks change to check for running ones
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === "running");
    if (hasRunning) {
      const interval = setInterval(fetchTasks, 3000);
      return () => clearInterval(interval);
    }
  }, [tasks]);

  const formatTime = (isoString) => {
    if (!isoString) return "N/A";
    return new Date(isoString).toLocaleString();
  };

  const getStatusBadge = (status) => {
    const variants = {
      pending: "secondary",
      running: "default",
      completed: "success",
      failed: "destructive",
      cancelled: "outline",
    };
    return variants[status] || "secondary";
  };

  if (loading && tasks.length === 0) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-muted-foreground">Loading tasks...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg border border-red-200 bg-red-50 dark:bg-red-900/20">
        <p className="text-red-600 dark:text-red-400">Error: {error}</p>
        <Button variant="outline" size="sm" onClick={fetchTasks} className="mt-2">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <ListTodo className="h-5 w-5" />
              Evaluation Tasks
            </CardTitle>
            <CardDescription>
              Background evaluation jobs and their status
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={fetchTasks}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {tasks.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <ListTodo className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No evaluation tasks yet</p>
            <p className="text-sm">Start an evaluation to see it here</p>
          </div>
        ) : (
          <div className="space-y-2">
            {tasks.map((task) => {
              const statusConfig = STATUS_ICONS[task.status] || STATUS_ICONS.pending;
              const StatusIcon = statusConfig.icon;

              return (
                <div
                  key={task.id}
                  className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors cursor-pointer"
                  onClick={() => {
                    if (task.status === "completed" && task.result_run_id) {
                      onViewResults?.(task.result_run_id);
                    } else {
                      onSelectTask?.(task.id);
                    }
                  }}
                >
                  <div className="flex items-center gap-3">
                    <StatusIcon
                      className={`h-5 w-5 ${statusConfig.color} ${
                        statusConfig.animate ? "animate-spin" : ""
                      }`}
                    />
                    <div>
                      <div className="font-medium text-sm">
                        {task.eval_dataset_name || "Quick Test"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {formatTime(task.created_at)}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    {task.status === "running" && (
                      <div className="text-sm text-muted-foreground">
                        {task.current_pair}/{task.total_pairs} (
                        {task.progress_percent.toFixed(0)}%)
                      </div>
                    )}
                    <Badge
                      variant={getStatusBadge(task.status)}
                      className="capitalize"
                    >
                      {task.status}
                    </Badge>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

EvaluationTaskList.propTypes = {
  onSelectTask: PropTypes.func,
  onViewResults: PropTypes.func,
  limit: PropTypes.number,
};

export default EvaluationTaskList;
