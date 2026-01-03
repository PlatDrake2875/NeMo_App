import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { Textarea } from "../../ui/textarea";
import { Switch } from "../../ui/switch";
import { Slider } from "../../ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
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
  Loader2,
  Plus,
  Play,
  RefreshCw,
  Save,
  Trash2,
  Upload,
  X,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";

export function EvaluationPage() {
  // Evaluation dataset state
  const [evalDatasets, setEvalDatasets] = useState([]);
  const [selectedEvalDataset, setSelectedEvalDataset] = useState(null);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [qaPairs, setQaPairs] = useState([
    { query: "", groundTruth: "" },
  ]);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);

  // Available collections for RAG
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // Hyperparameter configuration
  const [config, setConfig] = useState({
    collection: "",
    enableRag: true,
    enableColbert: true,
    topK: 5,
    temperature: 0.1,
  });

  // Evaluation state
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [expandedRows, setExpandedRows] = useState({});

  // Fetch evaluation datasets and collections on mount
  useEffect(() => {
    fetchEvalDatasets();
    fetchCollections();
  }, []);

  const fetchEvalDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`);
      if (response.ok) {
        const data = await response.json();
        setEvalDatasets(data);
      }
    } catch (err) {
      console.error("Error fetching evaluation datasets:", err);
    }
  };

  const fetchCollections = async () => {
    setCollectionsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/processed-datasets`);
      if (response.ok) {
        const data = await response.json();
        // API returns {count, datasets: [...]}
        const datasets = data.datasets || data;
        const completedDatasets = datasets.filter((d) => d.processing_status === "completed");
        setCollections(completedDatasets);
        // Set default collection if available
        if (completedDatasets.length > 0 && !config.collection) {
          setConfig((prev) => ({
            ...prev,
            collection: completedDatasets[0].collection_name,
          }));
        }
      }
    } catch (err) {
      console.error("Error fetching collections:", err);
    } finally {
      setCollectionsLoading(false);
    }
  };

  // Q&A pair management
  const addQaPair = () => {
    setQaPairs([...qaPairs, { query: "", groundTruth: "" }]);
  };

  const removeQaPair = (index) => {
    if (qaPairs.length > 1) {
      setQaPairs(qaPairs.filter((_, i) => i !== index));
    }
  };

  const updateQaPair = (index, field, value) => {
    const updated = [...qaPairs];
    updated[index][field] = value;
    setQaPairs(updated);
  };

  // Create evaluation dataset
  const createEvalDataset = async () => {
    if (!newDatasetName.trim()) {
      alert("Please enter a dataset name");
      return;
    }

    const validPairs = qaPairs.filter(
      (p) => p.query.trim() && p.groundTruth.trim()
    );
    if (validPairs.length === 0) {
      alert("Please add at least one Q&A pair with both query and ground truth");
      return;
    }

    setIsCreatingDataset(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newDatasetName.trim(),
          pairs: validPairs.map((p) => ({
            query: p.query.trim(),
            ground_truth: p.groundTruth.trim(),
          })),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setEvalDatasets([...evalDatasets, data]);
        setSelectedEvalDataset(data.id);
        setNewDatasetName("");
        setQaPairs([{ query: "", groundTruth: "" }]);
      } else {
        const error = await response.json();
        alert(`Error creating dataset: ${error.detail}`);
      }
    } catch (err) {
      console.error("Error creating evaluation dataset:", err);
      alert("Failed to create evaluation dataset");
    } finally {
      setIsCreatingDataset(false);
    }
  };

  // Run evaluation
  const runEvaluation = async () => {
    if (!selectedEvalDataset) {
      alert("Please select an evaluation dataset");
      return;
    }
    if (!config.collection) {
      alert("Please select a document collection");
      return;
    }

    setIsEvaluating(true);
    setEvaluationProgress(0);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eval_dataset_id: selectedEvalDataset,
          collection_name: config.collection,
          use_rag: config.enableRag,
          use_colbert: config.enableColbert,
          top_k: config.topK,
          temperature: config.temperature,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        const error = await response.json();
        alert(`Evaluation failed: ${error.detail}`);
      }
    } catch (err) {
      console.error("Error running evaluation:", err);
      alert("Failed to run evaluation");
    } finally {
      setIsEvaluating(false);
      setEvaluationProgress(100);
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
        <Button variant="outline" size="sm" onClick={fetchCollections}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left Column: Evaluation Dataset */}
        <div className="space-y-6">
          {/* Select Existing Dataset */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Evaluation Dataset</CardTitle>
              <CardDescription>
                Select an existing dataset or create a new one with Q&A pairs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Dataset</Label>
                <Select
                  value={selectedEvalDataset || ""}
                  onValueChange={setSelectedEvalDataset}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select evaluation dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {evalDatasets.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.pair_count} pairs)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              {/* Create New Dataset */}
              <div className="space-y-4">
                <h4 className="font-medium">Create New Dataset</h4>
                <div className="space-y-2">
                  <Label>Dataset Name</Label>
                  <Input
                    placeholder="e.g., colbert-test-set"
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Q&A Pairs</Label>
                    <Button variant="outline" size="sm" onClick={addQaPair}>
                      <Plus className="h-4 w-4 mr-1" />
                      Add Pair
                    </Button>
                  </div>

                  <div className="space-y-4 max-h-[400px] overflow-y-auto">
                    {qaPairs.map((pair, index) => (
                      <div
                        key={index}
                        className="relative border rounded-lg p-3 space-y-2"
                      >
                        <div className="absolute top-2 right-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => removeQaPair(index)}
                            disabled={qaPairs.length === 1}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                        <div className="space-y-1 pr-8">
                          <Label className="text-xs text-muted-foreground">
                            Question {index + 1}
                          </Label>
                          <Input
                            placeholder="Enter question..."
                            value={pair.query}
                            onChange={(e) =>
                              updateQaPair(index, "query", e.target.value)
                            }
                          />
                        </div>
                        <div className="space-y-1">
                          <Label className="text-xs text-muted-foreground">
                            Ground Truth
                          </Label>
                          <Textarea
                            placeholder="Enter expected answer..."
                            value={pair.groundTruth}
                            onChange={(e) =>
                              updateQaPair(index, "groundTruth", e.target.value)
                            }
                            rows={2}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <Button
                  className="w-full"
                  onClick={createEvalDataset}
                  disabled={isCreatingDataset}
                >
                  {isCreatingDataset ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Create Dataset
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Configuration */}
        <div className="space-y-6">
          {/* Hyperparameter Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Configuration</CardTitle>
              <CardDescription>
                Configure RAG pipeline settings for evaluation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Document Collection */}
              <div className="space-y-2">
                <Label>Document Collection</Label>
                <Select
                  value={config.collection}
                  onValueChange={(value) =>
                    setConfig((prev) => ({ ...prev, collection: value }))
                  }
                  disabled={collectionsLoading}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select collection..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rag_documents">
                      rag_documents (default)
                    </SelectItem>
                    {collections.map((col) => (
                      <SelectItem key={col.id} value={col.collection_name}>
                        {col.name} ({col.chunk_count} chunks)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              {/* RAG Toggle */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Enable RAG</Label>
                  <p className="text-xs text-muted-foreground">
                    Retrieve documents to augment responses
                  </p>
                </div>
                <Switch
                  checked={config.enableRag}
                  onCheckedChange={(checked) =>
                    setConfig((prev) => ({ ...prev, enableRag: checked }))
                  }
                />
              </div>

              {/* ColBERT Toggle */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Enable ColBERT Reranking</Label>
                  <p className="text-xs text-muted-foreground">
                    Two-stage retrieval with semantic reranking
                  </p>
                </div>
                <Switch
                  checked={config.enableColbert}
                  onCheckedChange={(checked) =>
                    setConfig((prev) => ({ ...prev, enableColbert: checked }))
                  }
                  disabled={!config.enableRag}
                />
              </div>

              <Separator />

              {/* Top-K Slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Top-K Results</Label>
                  <span className="text-sm text-muted-foreground">
                    {config.topK}
                  </span>
                </div>
                <Slider
                  value={[config.topK]}
                  onValueChange={([value]) =>
                    setConfig((prev) => ({ ...prev, topK: value }))
                  }
                  min={1}
                  max={20}
                  step={1}
                  disabled={!config.enableRag}
                />
                <p className="text-xs text-muted-foreground">
                  Number of documents to retrieve
                </p>
              </div>

              {/* Temperature Slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Temperature</Label>
                  <span className="text-sm text-muted-foreground">
                    {config.temperature.toFixed(1)}
                  </span>
                </div>
                <Slider
                  value={[config.temperature * 10]}
                  onValueChange={([value]) =>
                    setConfig((prev) => ({ ...prev, temperature: value / 10 }))
                  }
                  min={0}
                  max={10}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  LLM response randomness (0 = deterministic)
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Run Evaluation Button */}
          <Button
            className="w-full"
            size="lg"
            onClick={runEvaluation}
            disabled={isEvaluating || !selectedEvalDataset || !config.collection}
          >
            {isEvaluating ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Running Evaluation...
              </>
            ) : (
              <>
                <Play className="h-5 w-5 mr-2" />
                Run Evaluation
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Results Section */}
      {results && (
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
              <div className="grid gap-4 md:grid-cols-4">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">
                    Avg. Answer Relevancy
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.answer_relevancy || 0)}`}>
                    {((results.metrics?.answer_relevancy || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">
                    Avg. Faithfulness
                  </p>
                  <p className={`text-2xl font-bold ${getScoreColor(results.metrics?.faithfulness || 0)}`}>
                    {((results.metrics?.faithfulness || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-1">
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
              <CardTitle className="text-lg">Detailed Results</CardTitle>
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
                    <TableHead className="text-right">Score</TableHead>
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
                      </TableRow>
                      {expandedRows[index] && (
                        <TableRow key={`${index}-expanded`}>
                          <TableCell colSpan={5} className="bg-muted/30">
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
                  Collection: {results.config?.collection || config.collection}
                </Badge>
                <Badge variant={results.config?.use_rag ? "default" : "secondary"}>
                  RAG: {results.config?.use_rag ? "ON" : "OFF"}
                </Badge>
                <Badge variant={results.config?.use_colbert ? "default" : "secondary"}>
                  ColBERT: {results.config?.use_colbert ? "ON" : "OFF"}
                </Badge>
                <Badge variant="outline">
                  Top-K: {results.config?.top_k || config.topK}
                </Badge>
                <Badge variant="outline">
                  Temperature: {results.config?.temperature || config.temperature}
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
