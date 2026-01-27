import { useEffect, useState, Fragment } from "react";
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
  Download,
  ArrowLeft,
  Settings,
  BarChart3,
  FileText,
  Zap,
  Target,
  Clock,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";
import { EvaluationConfigModal } from "./EvaluationConfigModal";
import { EvaluationJobsPanel } from "./EvaluationJobsPanel";

export function EvaluationPage() {
  // Results state
  const [results, setResults] = useState(null);
  const [expandedRows, setExpandedRows] = useState({});

  // Current run state
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

  // Load a past evaluation run
  const loadEvaluationRun = async (runId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`);
      if (response.ok) {
        const data = await response.json();
        setResults({
          name: data.name,  // Experiment name
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
      ) : null}

      {/* New Evaluation Modal */}
      <EvaluationConfigModal
        open={showConfigModal}
        onOpenChange={setShowConfigModal}
        onStartEvaluation={handleStartEvaluation}
      />

      {/* Results Section - Only shown in results view mode */}
      {viewMode === "results" && results && (
        <div className="space-y-6">
          {/* Results Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setViewMode("jobs");
                  setResults(null);
                  setCurrentRunId(null);
                }}
                className="gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </Button>
              <div className="h-6 w-px bg-border" />
              <div>
                <h2 className="text-xl font-semibold">{results.name || "Experiment Results"}</h2>
                <p className="text-sm text-muted-foreground">
                  {results.results?.length || 0} test pairs evaluated
                </p>
              </div>
            </div>
            {currentRunId && (
              <Button variant="outline" size="sm" onClick={() => exportRunAsCSV(currentRunId)}>
                <Download className="h-4 w-4 mr-2" />
                Export CSV
              </Button>
            )}
          </div>

          {/* Two-column layout: Config + Metrics */}
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Experiment Configuration */}
            <Card className="lg:col-span-1">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">Collection</span>
                    <span className="text-sm font-medium truncate max-w-[150px]">{results.config?.collection || "N/A"}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">LLM</span>
                    <span className="text-sm font-medium font-mono">{(results.config?.llm_model || "Unknown").split("/").pop()}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">Embedder</span>
                    <span className="text-sm font-medium font-mono">{(results.config?.embedder || "N/A").split("/").pop()}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">RAG</span>
                    <Badge variant={results.config?.use_rag ? "default" : "secondary"}>
                      {results.config?.use_rag ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">Reranker</span>
                    <Badge variant={results.config?.use_colbert ? "default" : "outline"}>
                      {results.config?.use_colbert ? "ColBERT" : "None"}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-sm text-muted-foreground">Top-K</span>
                    <span className="text-sm font-medium">{results.config?.top_k ?? "N/A"}</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-sm text-muted-foreground">Temperature</span>
                    <span className="text-sm font-medium">{results.config?.temperature ?? "N/A"}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Metrics Summary - Enhanced */}
            <Card className="lg:col-span-2">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Performance Metrics
                </CardTitle>
                <CardDescription>Aggregate scores across all test queries</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {/* Correctness */}
                  <div className="relative p-4 rounded-xl bg-gradient-to-br from-blue-50 to-blue-100/50 dark:from-blue-950/30 dark:to-blue-900/20 border border-blue-200/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="h-4 w-4 text-blue-600" />
                      <span className="text-sm font-medium text-blue-900 dark:text-blue-100">Correctness</span>
                    </div>
                    <p className={`text-3xl font-bold ${getScoreColor(results.metrics?.answer_correctness || 0)}`}>
                      {((results.metrics?.answer_correctness || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">F1 factual + semantic similarity</p>
                  </div>

                  {/* Faithfulness */}
                  <div className="relative p-4 rounded-xl bg-gradient-to-br from-green-50 to-green-100/50 dark:from-green-950/30 dark:to-green-900/20 border border-green-200/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="h-4 w-4 text-green-600" />
                      <span className="text-sm font-medium text-green-900 dark:text-green-100">Faithfulness</span>
                    </div>
                    <p className={`text-3xl font-bold ${getScoreColor(results.metrics?.faithfulness || 0)}`}>
                      {((results.metrics?.faithfulness || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Claims supported by context</p>
                  </div>

                  {/* Relevancy */}
                  <div className="relative p-4 rounded-xl bg-gradient-to-br from-purple-50 to-purple-100/50 dark:from-purple-950/30 dark:to-purple-900/20 border border-purple-200/50">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="h-4 w-4 text-purple-600" />
                      <span className="text-sm font-medium text-purple-900 dark:text-purple-100">Relevancy</span>
                    </div>
                    <p className={`text-3xl font-bold ${getScoreColor(results.metrics?.answer_relevancy || 0)}`}>
                      {((results.metrics?.answer_relevancy || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Query-answer embedding similarity</p>
                  </div>

                  {/* Context Precision */}
                  <div className="relative p-4 rounded-xl bg-gradient-to-br from-orange-50 to-orange-100/50 dark:from-orange-950/30 dark:to-orange-900/20 border border-orange-200/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="h-4 w-4 text-orange-600" />
                      <span className="text-sm font-medium text-orange-900 dark:text-orange-100">Context Precision</span>
                    </div>
                    <p className={`text-3xl font-bold ${getScoreColor(results.metrics?.context_precision || 0)}`}>
                      {((results.metrics?.context_precision || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Retrieval ranking quality</p>
                  </div>

                  {/* Latency */}
                  <div className="relative p-4 rounded-xl bg-gradient-to-br from-gray-50 to-gray-100/50 dark:from-gray-950/30 dark:to-gray-900/20 border border-gray-200/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="h-4 w-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">Avg. Latency</span>
                    </div>
                    <p className="text-3xl font-bold text-gray-700 dark:text-gray-300">
                      {(results.metrics?.avg_latency || 0).toFixed(2)}s
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">Response time per query</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results Table */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Detailed Results
              </CardTitle>
              <CardDescription>
                Click on a row to expand and see retrieved context
              </CardDescription>
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
                    <Fragment key={index}>
                      <TableRow
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
                    </Fragment>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

        </div>
      )}
    </div>
  );
}

export default EvaluationPage;
