import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../ui/table";
import { Badge } from "../../ui/badge";
import { Separator } from "../../ui/separator";
import {
  ChevronDown,
  ChevronRight,
  FlaskConical,
  RefreshCw,
  Trash2,
  Download,
  History,
  ArrowLeft,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";
import { EvaluationConfigModal } from "./EvaluationConfigModal";
import { EvaluationJobsPanel } from "./EvaluationJobsPanel";

export function EvaluationPage() {
  // Results state
  const [results, setResults] = useState(null);
  const [expandedRows, setExpandedRows] = useState({});

  // Evaluation history state
  const [evaluationRuns, setEvaluationRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [currentRunId, setCurrentRunId] = useState(null);

  // Background task state
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [activeTasks, setActiveTasks] = useState([]);

  // UI state
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [viewMode, setViewMode] = useState("jobs"); // "jobs" or "results"

  // Fetch active/running tasks
  const fetchActiveTasks = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/tasks?limit=10`);
      if (response.ok) {
        const tasks = await response.json();
        setActiveTasks(tasks);
        // If there's a running task and we don't have one selected, select it
        const runningTask = tasks.find(t => t.status === "running");
        if (runningTask && !currentTaskId) {
          setCurrentTaskId(runningTask.id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch active tasks:", err);
    }
  };

  // Auto-refresh active tasks when there are running ones
  useEffect(() => {
    fetchActiveTasks();

    const interval = setInterval(() => {
      const hasRunning = activeTasks.some(t => t.status === "running" || t.status === "pending");
      if (hasRunning) {
        fetchActiveTasks();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [activeTasks.length]);

  // Fetch evaluation runs on mount
  useEffect(() => {
    fetchEvaluationRuns();
  }, []);

  // Fetch past evaluation runs
  const fetchEvaluationRuns = async () => {
    setRunsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs`);
      if (response.ok) {
        const data = await response.json();
        setEvaluationRuns(data);
      }
    } catch (err) {
      console.error("Error fetching evaluation runs:", err);
    } finally {
      setRunsLoading(false);
    }
  };

  // Load a past evaluation run
  const loadEvaluationRun = async (runId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`);
      if (response.ok) {
        const data = await response.json();
        setResults({
          results: data.results,
          metrics: data.metrics,
          config: data.config,
        });
        setCurrentRunId(runId);
        setExpandedRows({});
      }
    } catch (err) {
      console.error("Error loading evaluation run:", err);
      alert("Failed to load evaluation run");
    }
  };

  // Export evaluation run as CSV
  const exportRunAsCSV = (runId) => {
    window.open(`${API_BASE_URL}/api/evaluation/runs/${runId}/csv`, "_blank");
  };

  // Delete an evaluation run
  const deleteEvaluationRun = async (runId) => {
    if (!confirm("Are you sure you want to delete this evaluation run?")) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`, {
        method: "DELETE",
      });
      if (response.ok) {
        fetchEvaluationRuns();
        if (currentRunId === runId) {
          setResults(null);
          setCurrentRunId(null);
        }
      }
    } catch (err) {
      console.error("Error deleting evaluation run:", err);
    }
  };

  // Toggle row expansion
  const toggleRowExpansion = (index) => {
    setExpandedRows((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // Get score color
  const getScoreColor = (score) => {
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  // Get score badge variant
  const getScoreBadge = (score) => {
    if (score >= 0.8) return "default";
    if (score >= 0.5) return "secondary";
    return "destructive";
  };

  // Handle starting evaluation from modal
  const handleStartEvaluation = (taskId) => {
    setCurrentTaskId(taskId);
    fetchActiveTasks();
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <FlaskConical className="h-6 w-6" />
            RAG Evaluation
          </h2>
          <p className="text-muted-foreground">
            Evaluate your RAG pipeline with Q&A test sets and compare different configurations
          </p>
        </div>
      </div>

      {/* Main Content - Jobs Panel or Results View */}
      {viewMode === "jobs" ? (
        <div className="space-y-6">
          {/* Evaluation Jobs Panel with Tabs */}
          <EvaluationJobsPanel
            onViewResults={(runId) => {
              loadEvaluationRun(runId);
              setViewMode("results");
            }}
            onNewEvaluation={() => setShowConfigModal(true)}
          />
        </div>
      ) : (
        /* Results View */
        <div className="space-y-6">
          {/* Back to Jobs button */}
          <Button
            variant="outline"
            onClick={() => {
              setViewMode("jobs");
              setResults(null);
              setCurrentRunId(null);
            }}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Jobs
          </Button>

          {/* Evaluation History - Compact view when in results mode */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <History className="h-5 w-5" />
                  <CardTitle className="text-lg">Evaluation History</CardTitle>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchEvaluationRuns}
                  disabled={runsLoading}
                >
                  <RefreshCw className={`h-4 w-4 mr-1 ${runsLoading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
              </div>
              <CardDescription>
                Select an evaluation run to view detailed results
              </CardDescription>
            </CardHeader>
            <CardContent>
              {evaluationRuns.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No evaluation runs yet.
                </p>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {evaluationRuns.map((run) => (
                    <div
                      key={run.id}
                      className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                        currentRunId === run.id ? "border-primary bg-primary/5" : "hover:bg-muted/50"
                      }`}
                      onClick={() => loadEvaluationRun(run.id)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium truncate">{run.name}</span>
                          <Badge variant="outline" className="text-xs">
                            {run.pair_count} pairs
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                          <span>{new Date(run.created_at).toLocaleString()}</span>
                          {run.config?.llm_model && (
                            <span className="font-mono">{run.config.llm_model.split("/").pop()}</span>
                          )}
                          {run.metrics && (
                            <>
                              <span>Correctness: {((run.metrics.answer_correctness || 0) * 100).toFixed(0)}%</span>
                              <span>Faithful: {((run.metrics.faithfulness || 0) * 100).toFixed(0)}%</span>
                            </>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-1 ml-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            exportRunAsCSV(run.id);
                          }}
                          title="Export as CSV"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteEvaluationRun(run.id);
                          }}
                          title="Delete"
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* New Evaluation Modal */}
      <EvaluationConfigModal
        open={showConfigModal}
        onOpenChange={setShowConfigModal}
        onStartEvaluation={handleStartEvaluation}
      />

      {/* Results Section - Only shown in results view mode */}
      {viewMode === "results" && results && (
        <div className="space-y-6">
          <Separator />

          {/* Metrics Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Evaluation Metrics</CardTitle>
              <CardDescription>
                Aggregate scores across all test queries
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-5">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1" title="F1 factual similarity + semantic similarity">
                    Avg. Answer Correctness
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.answer_correctness || 0)}`}>
                    {((results.metrics?.answer_correctness || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1" title="Claims in answer supported by retrieved context">
                    Avg. Faithfulness
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.faithfulness || 0)}`}>
                    {((results.metrics?.faithfulness || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1" title="Embedding similarity between query and answer">
                    Avg. Answer Relevancy
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.answer_relevancy || 0)}`}>
                    {((results.metrics?.answer_relevancy || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1" title="Retrieval ranking quality (mean precision@k)">
                    Avg. Context Precision
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.context_precision || 0)}`}>
                    {((results.metrics?.context_precision || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">
                    Avg. Latency
                  </p>
                  <p className="text-2xl font-bold">
                    {(results.metrics?.avg_latency || 0).toFixed(2)}s
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results Table */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Detailed Results</CardTitle>
                  <CardDescription>
                    Click on a row to expand and see retrieved context
                  </CardDescription>
                </div>
                {currentRunId && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => exportRunAsCSV(currentRunId)}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export CSV
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-8"></TableHead>
                    <TableHead>Query</TableHead>
                    <TableHead>Predicted Answer</TableHead>
                    <TableHead>Ground Truth</TableHead>
                    <TableHead className="text-right" title="Jaccard: Word overlap between predicted and ground truth">Jaccard</TableHead>
                    <TableHead className="text-right" title="Answer Correctness: F1 factual + semantic similarity">Correctness</TableHead>
                    <TableHead className="text-right" title="Faithfulness: Claims supported by retrieved context">Faithful</TableHead>
                    <TableHead className="text-right" title="Context Precision: Retrieval ranking quality">Ctx Prec</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.results?.map((result, index) => (
                    <>
                      <TableRow
                        key={index}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => toggleRowExpansion(index)}
                      >
                        <TableCell>
                          {expandedRows[index] ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </TableCell>
                        <TableCell className="max-w-[200px] truncate font-medium">
                          {result.query}
                        </TableCell>
                        <TableCell className="max-w-[250px] truncate">
                          {result.predicted_answer}
                        </TableCell>
                        <TableCell className="max-w-[250px] truncate text-muted-foreground">
                          {result.ground_truth}
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={getScoreBadge(result.score || 0)}>
                            {((result.score || 0) * 100).toFixed(0)}%
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={getScoreBadge(result.scores?.answer_correctness || 0)}>
                            {((result.scores?.answer_correctness || 0) * 100).toFixed(0)}%
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={getScoreBadge(result.scores?.faithfulness || 0)}>
                            {((result.scores?.faithfulness || 0) * 100).toFixed(0)}%
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={getScoreBadge(result.scores?.context_precision || 0)}>
                            {((result.scores?.context_precision || 0) * 100).toFixed(0)}%
                          </Badge>
                        </TableCell>
                      </TableRow>
                      {expandedRows[index] && (
                        <TableRow key={`${index}-expanded`}>
                          <TableCell colSpan={8} className="bg-muted/30">
                            <div className="p-4 space-y-4">
                              <div>
                                <h4 className="font-medium mb-2">Full Query</h4>
                                <p className="text-sm">{result.query}</p>
                              </div>
                              <div>
                                <h4 className="font-medium mb-2">Predicted Answer</h4>
                                <p className="text-sm whitespace-pre-wrap">
                                  {result.predicted_answer}
                                </p>
                              </div>
                              <div>
                                <h4 className="font-medium mb-2">Ground Truth</h4>
                                <p className="text-sm whitespace-pre-wrap text-muted-foreground">
                                  {result.ground_truth}
                                </p>
                              </div>
                              {result.retrieved_chunks && (
                                <div>
                                  <h4 className="font-medium mb-2">
                                    Retrieved Chunks ({result.retrieved_chunks.length})
                                  </h4>
                                  <div className="space-y-2">
                                    {result.retrieved_chunks.map((chunk, chunkIndex) => (
                                      <div
                                        key={chunkIndex}
                                        className="text-xs bg-background p-2 rounded border"
                                      >
                                        <p className="text-muted-foreground mb-1">
                                          Source: {chunk.source || "Unknown"}
                                        </p>
                                        <p className="line-clamp-3">{chunk.content}</p>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              <div className="flex gap-4 text-sm">
                                <span>
                                  Latency: <strong>{(result.latency || 0).toFixed(2)}s</strong>
                                </span>
                                {result.scores && (
                                  <>
                                    <span>
                                      Relevancy:{" "}
                                      <strong className={getScoreColor(result.scores.relevancy || 0)}>
                                        {((result.scores.relevancy || 0) * 100).toFixed(0)}%
                                      </strong>
                                    </span>
                                    <span>
                                      Faithfulness:{" "}
                                      <strong className={getScoreColor(result.scores.faithfulness || 0)}>
                                        {((result.scores.faithfulness || 0) * 100).toFixed(0)}%
                                      </strong>
                                    </span>
                                  </>
                                )}
                              </div>
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Configuration Used */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Evaluation Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">
                  Collection: {results.config?.collection || "N/A"}
                </Badge>
                <Badge variant={results.config?.use_rag ? "default" : "secondary"}>
                  RAG: {results.config?.use_rag ? "ON" : "OFF"}
                </Badge>
                <Badge variant="outline">
                  LLM: {(results.config?.llm_model || "Unknown").split("/").pop()}
                </Badge>
                <Badge variant="outline">
                  Embedder: {(results.config?.embedder || "N/A").split("/").pop()}
                </Badge>
                <Badge variant={results.config?.use_colbert ? "default" : "secondary"}>
                  Reranker: {results.config?.use_colbert ? "ColBERT" : "None"}
                </Badge>
                <Badge variant="outline">
                  Top-K: {results.config?.top_k ?? "N/A"}
                </Badge>
                <Badge variant="outline">
                  Temperature: {results.config?.temperature ?? "N/A"}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default EvaluationPage;
