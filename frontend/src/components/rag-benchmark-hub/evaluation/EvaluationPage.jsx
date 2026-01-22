import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Label } from "../../ui/label";
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
  Play,
  RefreshCw,
  AlertCircle,
  Database,
  Trash2,
  Download,
  History,
  Eye,
  ListTodo,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";
import { EvaluationTaskProgress } from "./EvaluationTaskProgress";
import { EvaluationTaskList } from "./EvaluationTaskList";

// Available embedding models
const AVAILABLE_EMBEDDERS = [
  { value: "sentence-transformers/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2", description: "Default, Fast (384d)" },
  { value: "sentence-transformers/all-mpnet-base-v2", label: "all-mpnet-base-v2", description: "Higher accuracy (768d)" },
  { value: "BAAI/bge-small-en-v1.5", label: "bge-small-en-v1.5", description: "BGE Small (384d)" },
  { value: "BAAI/bge-base-en-v1.5", label: "bge-base-en-v1.5", description: "BGE Base (768d)" },
  { value: "nomic-ai/nomic-embed-text-v1", label: "nomic-embed-text-v1", description: "Nomic, Long context (768d)" },
];

// Available reranking strategies
const AVAILABLE_RERANKERS = [
  { value: "none", label: "None", description: "No reranking, vector similarity only" },
  { value: "colbert", label: "ColBERT", description: "Two-stage retrieval with semantic reranking" },
];

export function EvaluationPage() {

  // Available collections for RAG
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // Hyperparameter configuration
  const [config, setConfig] = useState({
    collection: "",
    enableRag: true,
    embedder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "colbert",
    enableReranking: true,
    topK: 5,
    temperature: 0.1,
  });

  // Evaluation state
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationProgress, setEvaluationProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [expandedRows, setExpandedRows] = useState({});

  // Auto-creation state for new collections
  const [isCreatingCollection, setIsCreatingCollection] = useState(false);
  const [creationStatus, setCreationStatus] = useState("");
  const [pendingEmbedderCreation, setPendingEmbedderCreation] = useState(null);

  // Evaluation dataset state
  const [evalDatasets, setEvalDatasets] = useState([]);
  const [selectedEvalDataset, setSelectedEvalDataset] = useState("none");

  // Evaluation history state
  const [evaluationRuns, setEvaluationRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [currentRunId, setCurrentRunId] = useState(null);

  // Background task state
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [showTaskList, setShowTaskList] = useState(false);
  const [activeTasks, setActiveTasks] = useState([]);
  const [activeTasksLoading, setActiveTasksLoading] = useState(false);

  // Fetch active/running tasks
  const fetchActiveTasks = async () => {
    try {
      setActiveTasksLoading(true);
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
    } finally {
      setActiveTasksLoading(false);
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

  // Fetch collections, eval datasets, and evaluation runs on mount
  useEffect(() => {
    fetchCollections();
    fetchEvalDatasets();
    fetchEvaluationRuns();
  }, []);

  // Fetch evaluation datasets
  const fetchEvalDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`);
      if (response.ok) {
        const data = await response.json();
        setEvalDatasets(data);
      }
    } catch (err) {
      console.error("Error fetching eval datasets:", err);
    }
  };

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

  // Delete evaluation dataset
  const deleteEvalDataset = async (datasetId) => {
    if (!confirm("Are you sure you want to delete this evaluation dataset?")) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/${datasetId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        await fetchEvalDatasets();
        if (selectedEvalDataset === datasetId) {
          setSelectedEvalDataset("");
        }
      }
    } catch (err) {
      console.error("Error deleting eval dataset:", err);
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

  // Run evaluation (creates collection first if embedder mismatch)
  const runEvaluation = async () => {
    if (!config.collection) {
      alert("Please select a document collection");
      return;
    }

    setIsEvaluating(true);
    setEvaluationProgress(0);
    setResults(null);

    let collectionToUse = config.collection;

    try {
      // Check if we need to create a new collection first
      const mismatch = getEmbedderMismatch();
      const alternative = findAlternativeCollection();

      if (mismatch && !alternative) {
        // Need to create new collection with selected embedder
        const selectedCol = getSelectedCollection();
        if (selectedCol) {
          setCreationStatus("Creating collection with new embedder...");
          const newCollection = await autoCreateCollectionAndWait(selectedCol, config.embedder);
          if (newCollection) {
            collectionToUse = newCollection.collection_name;
            // Update config to use new collection
            setConfig(prev => ({ ...prev, collection: collectionToUse }));
          } else {
            throw new Error("Failed to create collection with selected embedder");
          }
        }
      } else if (mismatch && alternative) {
        // Use the alternative collection
        collectionToUse = alternative.collection_name;
        setConfig(prev => ({ ...prev, collection: collectionToUse }));
      }

      setCreationStatus("");

      // Start evaluation as background task
      const response = await fetch(`${API_BASE_URL}/api/evaluation/tasks/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eval_dataset_id: selectedEvalDataset === "none" ? null : selectedEvalDataset,
          collection_name: collectionToUse,
          use_rag: config.enableRag,
          embedder: config.embedder,
          reranker: config.reranker,
          use_colbert: config.reranker === "colbert" && config.enableReranking,
          top_k: config.topK,
          temperature: config.temperature,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentTaskId(data.task_id);
        // Task started - progress will be shown via EvaluationTaskProgress component
      } else {
        const error = await response.json();
        const errorMsg = error.detail || error.message || JSON.stringify(error);
        alert(`Failed to start evaluation: ${errorMsg}`);
        setIsEvaluating(false);
      }
    } catch (err) {
      console.error("Error starting evaluation:", err);
      alert(`Failed to start evaluation: ${err.message}`);
      setIsEvaluating(false);
    } finally {
      setCreationStatus("");
    }
  };

  // Handle task completion
  const handleTaskComplete = (task) => {
    setIsEvaluating(false);
    setCurrentTaskId(null);
    fetchEvaluationRuns();
    if (task.result_run_id) {
      loadEvaluationRun(task.result_run_id);
    }
  };

  // Handle viewing results from task list
  const handleViewResults = (runId) => {
    loadEvaluationRun(runId);
  };

  // Create collection and wait for completion (returns the new collection or null)
  const autoCreateCollectionAndWait = async (sourceCollection, newEmbedder) => {
    if (!sourceCollection || !sourceCollection.raw_dataset_id) {
      console.error("Cannot create collection: no source raw dataset");
      return null;
    }

    setIsCreatingCollection(true);
    setPendingEmbedderCreation(newEmbedder);

    try {
      const embedderShort = newEmbedder.split("/").pop().replace(/-/g, "_");
      const newName = `${sourceCollection.name}_${embedderShort}`;

      setCreationStatus("Initializing dataset...");
      const createResponse = await fetch(`${API_BASE_URL}/api/processed-datasets?start_processing=true`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newName,
          description: `Auto-created from ${sourceCollection.name} with ${embedderShort}`,
          raw_dataset_id: sourceCollection.raw_dataset_id,
          vector_backend: sourceCollection.vector_backend || "pgvector",
          embedder_config: {
            model_name: newEmbedder,
            model_type: "huggingface",
            model_kwargs: newEmbedder.includes("nomic") ? { trust_remote_code: true } : {},
          },
          preprocessing_config: sourceCollection.preprocessing_config || {
            cleaning: { enabled: false },
            llm_metadata: { enabled: false },
            chunking: {
              method: "recursive",
              chunk_size: 1000,
              chunk_overlap: 200,
            },
          },
        }),
      });

      if (!createResponse.ok) {
        const error = await createResponse.json();
        throw new Error(error.detail || "Failed to create dataset");
      }

      const newDataset = await createResponse.json();
      setCreationStatus("Processing documents...");

      // Poll for completion
      let completed = false;
      let attempts = 0;
      const maxAttempts = 300; // 5 minutes max

      while (!completed && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;

        const statusResponse = await fetch(`${API_BASE_URL}/api/processed-datasets/${newDataset.id}/status`);
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          setCreationStatus(`Processing... ${status.current_step || ""}`);

          if (status.status === "completed") {
            completed = true;
          } else if (status.status === "failed") {
            throw new Error(status.error || "Processing failed");
          }
        }
      }

      if (!completed) {
        throw new Error("Processing timed out");
      }

      // Refresh collections list
      await fetchCollections();

      return newDataset;

    } catch (err) {
      console.error("Error creating collection:", err);
      throw err;
    } finally {
      setIsCreatingCollection(false);
      setPendingEmbedderCreation(null);
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

  // Get the selected collection's full info
  const getSelectedCollection = () => {
    if (!config.collection || config.collection === "rag_documents") return null;
    return collections.find(c => c.collection_name === config.collection);
  };

  // Normalize embedder name for comparison (handle prefix variations)
  const normalizeEmbedderName = (name) => {
    if (!name) return "";
    // Extract just the model name (last part after /)
    return name.split("/").pop().toLowerCase();
  };

  // Check if selected embedder matches the collection's embedder
  const getEmbedderMismatch = () => {
    const selectedCol = getSelectedCollection();
    if (!selectedCol) return null;

    const collectionEmbedder = selectedCol.embedder_config?.model_name;
    if (!collectionEmbedder) return null;

    // Compare normalized names (handles "all-MiniLM-L6-v2" vs "sentence-transformers/all-MiniLM-L6-v2")
    const normalizedCollection = normalizeEmbedderName(collectionEmbedder);
    const normalizedSelected = normalizeEmbedderName(config.embedder);

    if (normalizedCollection !== normalizedSelected) {
      return {
        collectionEmbedder,
        selectedEmbedder: config.embedder,
        collectionName: selectedCol.name,
      };
    }
    return null;
  };

  // Find alternative collection with the selected embedder (same raw dataset)
  const findAlternativeCollection = () => {
    const selectedCol = getSelectedCollection();
    if (!selectedCol) return null;

    const normalizedSelected = normalizeEmbedderName(config.embedder);

    return collections.find(col =>
      col.id !== selectedCol.id &&
      col.raw_dataset_id === selectedCol.raw_dataset_id &&
      normalizeEmbedderName(col.embedder_config?.model_name) === normalizedSelected &&
      col.processing_status === "completed"
    );
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

      <div className="max-w-2xl mx-auto space-y-6">
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
                        <div className="flex items-center gap-2">
                          <span>{col.name}</span>
                          <span className="text-muted-foreground">
                            ({col.chunk_count} chunks)
                          </span>
                          {col.config_hash && (
                            <Badge variant="outline" className="text-[10px] px-1 py-0 font-mono">
                              {col.config_hash}
                            </Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {/* Show selected collection's embedder info */}
                {config.collection && config.collection !== "rag_documents" && (
                  <p className="text-xs text-muted-foreground">
                    Indexed with: {
                      collections.find(c => c.collection_name === config.collection)?.embedder_config?.model_name?.split("/").pop() || "unknown"
                    }
                  </p>
                )}
              </div>

              {/* === RAG Activation Section === */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Separator className="flex-1" />
                  <span className="text-xs text-muted-foreground font-medium px-2">RAG Activation</span>
                  <Separator className="flex-1" />
                </div>

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
              </div>

              {/* === Embedders & Rerankers Section === */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Separator className="flex-1" />
                  <span className="text-xs text-muted-foreground font-medium px-2">Embedders & Rerankers</span>
                  <Separator className="flex-1" />
                </div>

                {/* Embedding Model Select */}
                <div className="space-y-2">
                  <Label>Embedding Model</Label>
                  <Select
                    value={config.embedder}
                    onValueChange={(value) =>
                      setConfig((prev) => ({ ...prev, embedder: value }))
                    }
                    disabled={!config.enableRag}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select embedder..." />
                    </SelectTrigger>
                    <SelectContent>
                      {AVAILABLE_EMBEDDERS.map((emb) => (
                        <SelectItem key={emb.value} value={emb.value}>
                          {emb.label} - {emb.description}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Converts text to vectors for similarity search
                  </p>
                  {/* Embedder mismatch warning */}
                  {(() => {
                    const mismatch = getEmbedderMismatch();
                    const alternative = findAlternativeCollection();

                    if (!mismatch) return null;

                    return (
                      <div className="space-y-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded-md">
                        <div className="flex items-start gap-2 text-xs text-yellow-600 dark:text-yellow-500">
                          <AlertCircle className="h-3 w-3 mt-0.5 flex-shrink-0" />
                          <div>
                            <p className="font-medium">Embedder mismatch</p>
                            <p>
                              "{mismatch.collectionName}" was indexed with{" "}
                              <span className="font-mono">{mismatch.collectionEmbedder.split("/").pop()}</span>
                            </p>
                          </div>
                        </div>

                        {alternative ? (
                          <Button
                            variant="outline"
                            size="sm"
                            className="w-full text-xs h-7"
                            onClick={() => setConfig(prev => ({ ...prev, collection: alternative.collection_name }))}
                          >
                            Use "{alternative.name}" ({alternative.config_hash})
                          </Button>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            A new collection will be created when you run evaluation.
                          </p>
                        )}
                      </div>
                    );
                  })()}
                </div>

                {/* Reranking Strategy Select */}
                <div className="space-y-2">
                  <Label>Reranking Strategy</Label>
                  <Select
                    value={config.reranker}
                    onValueChange={(value) =>
                      setConfig((prev) => ({ ...prev, reranker: value }))
                    }
                    disabled={!config.enableRag}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select reranker..." />
                    </SelectTrigger>
                    <SelectContent>
                      {AVAILABLE_RERANKERS.map((rr) => (
                        <SelectItem key={rr.value} value={rr.value}>
                          {rr.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Enable Reranking Toggle (only shown if reranker != "none") */}
                {config.reranker !== "none" && (
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Enable Reranking</Label>
                      <p className="text-xs text-muted-foreground">
                        {AVAILABLE_RERANKERS.find(r => r.value === config.reranker)?.description}
                      </p>
                    </div>
                    <Switch
                      checked={config.enableReranking}
                      onCheckedChange={(checked) =>
                        setConfig((prev) => ({ ...prev, enableReranking: checked }))
                      }
                      disabled={!config.enableRag}
                    />
                  </div>
                )}
              </div>

              {/* === Retrieval Settings Section === */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Separator className="flex-1" />
                  <span className="text-xs text-muted-foreground font-medium px-2">Retrieval Settings</span>
                  <Separator className="flex-1" />
                </div>

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
              </div>
            </CardContent>
          </Card>

          {/* Evaluation Dataset Selection Card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Database className="h-5 w-5" />
                Evaluation Dataset
              </CardTitle>
              <CardDescription>
                Select a Q&A dataset to evaluate against (optional)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Select
                  value={selectedEvalDataset}
                  onValueChange={setSelectedEvalDataset}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Quick test (no ground truth)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Quick test (no ground truth)</SelectItem>
                    {evalDatasets.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        <div className="flex items-center gap-2">
                          <span>{dataset.name}</span>
                          <span className="text-muted-foreground">
                            ({dataset.pair_count} pairs)
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedEvalDataset !== "none" && (
                  <div className="flex items-center justify-between p-2 bg-muted rounded-md">
                    <p className="text-xs text-muted-foreground">
                      {evalDatasets.find(d => d.id === selectedEvalDataset)?.pair_count || 0} Q&A pairs will be evaluated
                    </p>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-destructive"
                      onClick={() => deleteEvalDataset(selectedEvalDataset)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Run Evaluation Button */}
          <div className="flex gap-2">
            <Button
              className="flex-1"
              size="lg"
              onClick={runEvaluation}
              disabled={isEvaluating || !config.collection}
            >
              {isEvaluating && !currentTaskId ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  {creationStatus || "Starting..."}
                </>
              ) : (
                <>
                  <Play className="h-5 w-5 mr-2" />
                  Run Evaluation {selectedEvalDataset !== "none" ? `(${evalDatasets.find(d => d.id === selectedEvalDataset)?.pair_count || 0} pairs)` : "(Quick Test)"}
                </>
              )}
            </Button>
            <Button
              variant={showTaskList ? "default" : "outline"}
              size="lg"
              onClick={() => setShowTaskList(!showTaskList)}
              className="px-3 relative"
              title="View all tasks"
            >
              <ListTodo className="h-5 w-5" />
              {activeTasks.filter(t => t.status === "running" || t.status === "pending").length > 0 && (
                <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center animate-pulse">
                  {activeTasks.filter(t => t.status === "running" || t.status === "pending").length}
                </span>
              )}
            </Button>
          </div>

          {/* Active Running Tasks - Always visible when there are running tasks */}
          {activeTasks.filter(t => t.status === "running" || t.status === "pending").length > 0 && (
            <Card className="border-blue-200 bg-blue-50/50 dark:bg-blue-950/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                  Active Evaluations
                  <Badge variant="secondary" className="ml-auto">
                    {activeTasks.filter(t => t.status === "running" || t.status === "pending").length} running
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {activeTasks
                  .filter(t => t.status === "running" || t.status === "pending")
                  .map(task => (
                    <div
                      key={task.id}
                      className="p-3 rounded-lg border bg-background cursor-pointer hover:bg-muted/50 transition-colors"
                      onClick={() => setCurrentTaskId(task.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">{task.eval_dataset_name || "Quick Test"}</span>
                        <Badge variant="outline" className="text-xs">
                          {task.current_pair}/{task.total_pairs} ({task.progress_percent.toFixed(0)}%)
                        </Badge>
                      </div>
                      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-500"
                          style={{ width: `${task.progress_percent}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 truncate">
                        {task.current_step || "Processing..."}
                      </p>
                    </div>
                  ))}
              </CardContent>
            </Card>
          )}

          {/* Current Task Progress (detailed view when clicked) */}
          {currentTaskId && (
            <EvaluationTaskProgress
              taskId={currentTaskId}
              onComplete={(task) => {
                handleTaskComplete(task);
                fetchActiveTasks();
              }}
              onViewResults={handleViewResults}
            />
          )}

          {/* Task List (toggleable - shows all tasks including completed) */}
          {showTaskList && (
            <EvaluationTaskList
              onSelectTask={(taskId) => setCurrentTaskId(taskId)}
              onViewResults={handleViewResults}
              limit={10}
            />
          )}
      </div>

      {/* Evaluation History */}
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
            Past evaluation runs are saved automatically. Load or export as CSV.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {evaluationRuns.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No evaluation runs yet. Run an evaluation to see results here.
            </p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {evaluationRuns.map((run) => (
                <div
                  key={run.id}
                  className={`flex items-center justify-between p-3 rounded-lg border ${
                    currentRunId === run.id ? "border-primary bg-primary/5" : "hover:bg-muted/50"
                  }`}
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
                      onClick={() => loadEvaluationRun(run.id)}
                      title="Load results"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => exportRunAsCSV(run.id)}
                      title="Export as CSV"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteEvaluationRun(run.id)}
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
                  Collection: {results.config?.collection || config.collection}
                </Badge>
                <Badge variant={results.config?.use_rag ? "default" : "secondary"}>
                  RAG: {results.config?.use_rag ? "ON" : "OFF"}
                </Badge>
                <Badge variant="outline">
                  LLM: {(results.config?.llm_model || "Unknown").split("/").pop()}
                </Badge>
                <Badge variant="outline">
                  Embedder: {(results.config?.embedder || config.embedder).split("/").pop()}
                </Badge>
                <Badge variant={results.config?.use_colbert ? "default" : "secondary"}>
                  Reranker: {results.config?.use_colbert ? "ColBERT" : "None"}
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
