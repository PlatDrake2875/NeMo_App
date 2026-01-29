import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import { Button } from "../../ui/button";
import { Label } from "../../ui/label";
import { Input } from "../../ui/input";
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
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../../ui/collapsible";
import { Badge } from "../../ui/badge";
import { Separator } from "../../ui/separator";
import {
  Play,
  Loader2,
  Database,
  FlaskConical,
  AlertCircle,
  Settings2,
  Zap,
  Settings,
  Sparkles,
  ChevronDown,
  ChevronRight,
  HardDrive,
} from "lucide-react";
import { getApiBaseUrl } from "../../../lib/api-config";

// Experiment presets - match backend EXPERIMENT_PRESETS
const EXPERIMENT_PRESETS = {
  quick: {
    name: "Quick Test",
    description: "Fast iteration, basic settings",
    icon: Zap,
    config: {
      chunking: { method: "recursive", chunkSize: 500, chunkOverlap: 50 },
      embedder: "sentence-transformers/all-MiniLM-L6-v2",
      topK: 3,
      reranker: "none",
      temperature: 0.1,
    },
  },
  balanced: {
    name: "Balanced",
    description: "Good quality, reasonable speed",
    icon: Settings,
    config: {
      chunking: { method: "recursive", chunkSize: 1000, chunkOverlap: 200 },
      embedder: "sentence-transformers/all-MiniLM-L6-v2",
      topK: 5,
      reranker: "none",
      temperature: 0.1,
    },
  },
  high_quality: {
    name: "High Quality",
    description: "Best accuracy, slower",
    icon: Sparkles,
    config: {
      chunking: { method: "recursive", chunkSize: 1500, chunkOverlap: 300 },
      embedder: "BAAI/bge-small-en-v1.5",
      topK: 7,
      reranker: "colbert",
      temperature: 0.1,
    },
  },
};

// Available embedding models
const AVAILABLE_EMBEDDERS = [
  { value: "sentence-transformers/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2", description: "Default, Fast (384d)" },
  { value: "sentence-transformers/all-mpnet-base-v2", label: "all-mpnet-base-v2", description: "Higher accuracy (768d)" },
  { value: "BAAI/bge-small-en-v1.5", label: "bge-small-en-v1.5", description: "BGE Small (384d)" },
  { value: "BAAI/bge-base-en-v1.5", label: "bge-base-en-v1.5", description: "BGE Base (768d)" },
  { value: "nomic-ai/nomic-embed-text-v1", label: "nomic-embed-text-v1", description: "Nomic, Long context (768d)" },
];

// Chunking methods
const CHUNKING_METHODS = [
  { value: "recursive", label: "Recursive", description: "Natural boundaries" },
  { value: "fixed", label: "Fixed Size", description: "Uniform chunks" },
  { value: "semantic", label: "Semantic", description: "Meaning-based splits" },
];

export function EvaluationConfigModal({ open, onOpenChange, onStartEvaluation }) {
  // Legacy collections state (for backward compatibility)
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // New: Preprocessed datasets state (for new flow)
  const [preprocessedDatasets, setPreprocessedDatasets] = useState([]);
  const [preprocessedLoading, setPreprocessedLoading] = useState(false);

  // Evaluation datasets state
  const [evalDatasets, setEvalDatasets] = useState([]);

  // Data source mode: "preprocessed" (new) or "collection" (legacy)
  const [dataSourceMode, setDataSourceMode] = useState("preprocessed");

  // Selected preset
  const [selectedPreset, setSelectedPreset] = useState("balanced");

  // Show advanced options
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Configuration state - now includes chunking
  const [config, setConfig] = useState({
    // Data source
    collection: "",
    preprocessedDatasetId: null,
    // Chunking (for new flow)
    chunkingMethod: "recursive",
    chunkSize: 1000,
    chunkOverlap: 200,
    // RAG settings
    enableRag: true,
    embedder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "none",
    enableReranking: false,
    topK: 5,
    temperature: 0.1,
  });

  const [selectedEvalDataset, setSelectedEvalDataset] = useState("none");
  const [experimentName, setExperimentName] = useState("");
  const [isStarting, setIsStarting] = useState(false);

  // Cache info for the current configuration
  const [cacheInfo, setCacheInfo] = useState(null);

  // Fetch collections and datasets when modal opens, reset form
  useEffect(() => {
    if (open) {
      fetchCollections();
      fetchPreprocessedDatasets();
      fetchEvalDatasets();
      // Reset experiment name and apply default preset when modal opens
      setExperimentName("");
      applyPreset("balanced");
    }
  }, [open]);

  // Apply preset configuration
  const applyPreset = (presetKey) => {
    const preset = EXPERIMENT_PRESETS[presetKey];
    if (!preset) return;

    setSelectedPreset(presetKey);
    setConfig((prev) => ({
      ...prev,
      chunkingMethod: preset.config.chunking.method,
      chunkSize: preset.config.chunking.chunkSize,
      chunkOverlap: preset.config.chunking.chunkOverlap,
      embedder: preset.config.embedder,
      topK: preset.config.topK,
      reranker: preset.config.reranker,
      enableReranking: preset.config.reranker === "colbert",
      temperature: preset.config.temperature,
    }));
  };

  const fetchCollections = async () => {
    setCollectionsLoading(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/processed-datasets`);
      if (response.ok) {
        const data = await response.json();
        const datasets = data.datasets || [];
        // Legacy collections: have chunk_count > 0 (actually indexed to vector store)
        const legacyCollections = datasets.filter(
          (d) => d.processing_status === "completed" && d.chunk_count > 0
        );
        setCollections(legacyCollections);
        if (legacyCollections.length > 0 && !config.collection) {
          setConfig((prev) => ({ ...prev, collection: legacyCollections[0].collection_name }));
        }
      }
    } catch (err) {
      console.error("Failed to fetch collections:", err);
    } finally {
      setCollectionsLoading(false);
    }
  };

  const fetchPreprocessedDatasets = async () => {
    setPreprocessedLoading(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/processed-datasets`);
      if (response.ok) {
        const data = await response.json();
        const datasets = data.datasets || [];
        // New flow: datasets with document_count > 0 (preprocessed docs saved)
        // OR legacy datasets with chunk_count > 0 (can still use them with new chunking)
        const preprocessed = datasets.filter(
          (d) => d.processing_status === "completed" && (d.document_count > 0 || d.chunk_count > 0)
        );
        setPreprocessedDatasets(preprocessed);
        if (preprocessed.length > 0 && !config.preprocessedDatasetId) {
          setConfig((prev) => ({ ...prev, preprocessedDatasetId: preprocessed[0].id }));
        }
      }
    } catch (err) {
      console.error("Failed to fetch preprocessed datasets:", err);
    } finally {
      setPreprocessedLoading(false);
    }
  };

  const fetchEvalDatasets = async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/evaluation/datasets`);
      if (response.ok) {
        const data = await response.json();
        setEvalDatasets(data);
      }
    } catch (err) {
      console.error("Failed to fetch evaluation datasets:", err);
    }
  };

  const getSelectedCollection = () => {
    return collections.find((c) => c.collection_name === config.collection);
  };

  const getSelectedPreprocessedDataset = () => {
    return preprocessedDatasets.find((d) => d.id === config.preprocessedDatasetId);
  };

  const handleStart = async () => {
    // Validate based on mode
    if (dataSourceMode === "collection" && !config.collection) {
      alert("Please select a collection");
      return;
    }
    if (dataSourceMode === "preprocessed" && !config.preprocessedDatasetId) {
      alert("Please select a preprocessed dataset");
      return;
    }

    setIsStarting(true);
    try {
      // Build request body based on mode
      const requestBody = {
        experiment_name: experimentName.trim() || null,
        eval_dataset_id: selectedEvalDataset === "none" ? null : selectedEvalDataset,
        use_rag: config.enableRag,
        embedder: config.embedder,
        use_colbert: config.reranker === "colbert" && config.enableReranking,
        top_k: config.topK,
        temperature: config.temperature,
      };

      if (dataSourceMode === "collection") {
        // Legacy mode: use existing collection
        requestBody.collection_name = config.collection;
      } else {
        // New mode: use preprocessed dataset + chunking config
        requestBody.preprocessed_dataset_id = config.preprocessedDatasetId;
        requestBody.preset = selectedPreset;
        requestBody.chunking = {
          method: config.chunkingMethod,
          chunk_size: config.chunkSize,
          chunk_overlap: config.chunkOverlap,
        };
      }

      const response = await fetch(`${getApiBaseUrl()}/api/evaluation/tasks/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        onStartEvaluation?.(data.task_id);
        onOpenChange(false);
        // Reset form
        setExperimentName("");
      } else {
        const error = await response.json();
        alert(`Failed to start evaluation: ${error.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Failed to start evaluation:", err);
      alert(`Failed to start evaluation: ${err.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  const selectedCollection = getSelectedCollection();
  const selectedPreprocessedDataset = getSelectedPreprocessedDataset();
  const selectedDataset = evalDatasets.find((d) => d.id === selectedEvalDataset);

  // Check if start button should be disabled
  const isStartDisabled =
    isStarting ||
    (dataSourceMode === "collection" && !config.collection) ||
    (dataSourceMode === "preprocessed" && !config.preprocessedDatasetId);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5" />
            New Evaluation
          </DialogTitle>
          <DialogDescription>
            Configure and start a new RAG evaluation run
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Experiment Name */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              Experiment Name
            </Label>
            <Input
              placeholder="e.g., RAG-v2-topk10-colbert"
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              maxLength={200}
              className="font-mono"
            />
            <p className="text-xs text-muted-foreground">
              Optional. Give your experiment a memorable name for easy identification.
            </p>
          </div>

          <Separator />

          {/* Experiment Presets */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Experiment Preset</Label>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(EXPERIMENT_PRESETS).map(([key, preset]) => {
                const Icon = preset.icon;
                return (
                  <Button
                    key={key}
                    variant={selectedPreset === key ? "default" : "outline"}
                    className="h-auto py-2 flex flex-col items-center gap-1"
                    onClick={() => applyPreset(key)}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="text-xs font-medium">{preset.name}</span>
                    <span className="text-[10px] text-muted-foreground">
                      {preset.description}
                    </span>
                  </Button>
                );
              })}
            </div>
          </div>

          <Separator />

          {/* Data Source Mode Toggle */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4" />
              Data Source
            </Label>
            <div className="flex gap-2">
              <Button
                variant={dataSourceMode === "preprocessed" ? "default" : "outline"}
                size="sm"
                className="flex-1"
                onClick={() => setDataSourceMode("preprocessed")}
              >
                <HardDrive className="h-4 w-4 mr-1" />
                Preprocessed Dataset
              </Button>
              <Button
                variant={dataSourceMode === "collection" ? "default" : "outline"}
                size="sm"
                className="flex-1"
                onClick={() => setDataSourceMode("collection")}
              >
                <Database className="h-4 w-4 mr-1" />
                Existing Collection
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              {dataSourceMode === "preprocessed"
                ? "Use cleaned data with customizable chunking/embedding (recommended)"
                : "Use a pre-indexed collection (legacy mode)"}
            </p>
          </div>

          {/* Preprocessed Dataset Selection (new mode) */}
          {dataSourceMode === "preprocessed" && (
            <div className="space-y-3">
              <Label className="text-sm font-medium">Preprocessed Dataset</Label>
              <Select
                value={config.preprocessedDatasetId?.toString() || ""}
                onValueChange={(v) =>
                  setConfig((prev) => ({ ...prev, preprocessedDatasetId: parseInt(v) }))
                }
                disabled={preprocessedLoading}
              >
                <SelectTrigger>
                  <SelectValue
                    placeholder={preprocessedLoading ? "Loading..." : "Select dataset..."}
                  />
                </SelectTrigger>
                <SelectContent>
                  {preprocessedDatasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id.toString()}>
                      <div className="flex items-center gap-2">
                        <span>{ds.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {ds.document_count > 0
                            ? `${ds.document_count} docs`
                            : `${ds.chunk_count} chunks`}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedPreprocessedDataset && (
                <p className="text-xs text-muted-foreground">
                  {selectedPreprocessedDataset.document_count > 0
                    ? `${selectedPreprocessedDataset.document_count} cleaned documents ready`
                    : `${selectedPreprocessedDataset.chunk_count} chunks (legacy dataset)`}
                </p>
              )}
            </div>
          )}

          {/* Collection Selection (legacy mode) */}
          {dataSourceMode === "collection" && (
            <div className="space-y-3">
              <Label className="text-sm font-medium">Document Collection</Label>
              <Select
                value={config.collection}
                onValueChange={(v) => setConfig((prev) => ({ ...prev, collection: v }))}
                disabled={collectionsLoading}
              >
                <SelectTrigger>
                  <SelectValue
                    placeholder={collectionsLoading ? "Loading..." : "Select collection..."}
                  />
                </SelectTrigger>
                <SelectContent>
                  {collections.map((col) => (
                    <SelectItem key={col.collection_name} value={col.collection_name}>
                      <div className="flex items-center gap-2">
                        <span>{col.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {col.chunk_count} chunks
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {col.vector_backend}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedCollection && (
                <p className="text-xs text-muted-foreground">
                  Embedder: {selectedCollection.embedder_model_name?.split("/").pop()}
                </p>
              )}
            </div>
          )}

          <Separator />

          {/* Evaluation Dataset Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <FlaskConical className="h-4 w-4" />
              Q&A Evaluation Dataset
            </Label>
            <Select
              value={selectedEvalDataset}
              onValueChange={(value) => {
                setSelectedEvalDataset(value);
                // Auto-select the matching collection if dataset has source_collection
                const dataset = evalDatasets.find((d) => d.id === value);
                if (
                  dataset?.source_collection &&
                  collections.some((c) => c.collection_name === dataset.source_collection)
                ) {
                  setConfig((prev) => ({ ...prev, collection: dataset.source_collection }));
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select dataset..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">
                  <div className="flex items-center gap-2">
                    <span>Quick Test (1 question)</span>
                    <Badge variant="secondary" className="text-xs">
                      Fast
                    </Badge>
                  </div>
                </SelectItem>
                {evalDatasets.map((dataset) => (
                  <SelectItem key={dataset.id} value={dataset.id}>
                    <div className="flex items-center gap-2">
                      <span>{dataset.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {dataset.pair_count} pairs
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedDataset && (
              <p className="text-xs text-muted-foreground">
                {selectedDataset.pair_count} Q&A pairs
              </p>
            )}
          </div>

          <Separator />

          {/* RAG Settings */}
          <div className="space-y-4">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Settings2 className="h-4 w-4" />
              RAG Settings
            </Label>

            <div className="grid grid-cols-2 gap-4">
              {/* Enable RAG */}
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <Label className="text-sm">Use RAG</Label>
                  <p className="text-xs text-muted-foreground">Retrieve context</p>
                </div>
                <Switch
                  checked={config.enableRag}
                  onCheckedChange={(v) =>
                    setConfig((prev) => ({ ...prev, enableRag: v }))
                  }
                />
              </div>

              {/* Enable Reranking */}
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <Label className="text-sm">ColBERT Rerank</Label>
                  <p className="text-xs text-muted-foreground">Two-stage retrieval</p>
                </div>
                <Switch
                  checked={config.enableReranking}
                  onCheckedChange={(v) =>
                    setConfig((prev) => ({ ...prev, enableReranking: v, reranker: v ? "colbert" : "none" }))
                  }
                  disabled={!config.enableRag}
                />
              </div>
            </div>

            {/* Top-K */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-sm">Top-K Documents</Label>
                <span className="text-sm text-muted-foreground">{config.topK}</span>
              </div>
              <Slider
                value={[config.topK]}
                onValueChange={([v]) => setConfig((prev) => ({ ...prev, topK: v }))}
                min={1}
                max={20}
                step={1}
                disabled={!config.enableRag}
              />
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-sm">Temperature</Label>
                <span className="text-sm text-muted-foreground">
                  {config.temperature.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[config.temperature]}
                onValueChange={([v]) =>
                  setConfig((prev) => ({ ...prev, temperature: v }))
                }
                min={0}
                max={1}
                step={0.05}
              />
            </div>
          </div>

          <Separator />

          {/* Advanced Options (Collapsible) */}
          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-2">
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Settings2 className="h-4 w-4" />
                  Advanced Options
                </span>
                {showAdvanced ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-2">
              {/* Embedder Selection */}
              <div className="space-y-2">
                <Label className="text-sm">Embedder Model</Label>
                <Select
                  value={config.embedder}
                  onValueChange={(v) => setConfig((prev) => ({ ...prev, embedder: v }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {AVAILABLE_EMBEDDERS.map((emb) => (
                      <SelectItem key={emb.value} value={emb.value}>
                        <div className="flex items-center gap-2">
                          <span>{emb.label}</span>
                          <span className="text-xs text-muted-foreground">
                            ({emb.description})
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Chunking Configuration (only for preprocessed mode) */}
              {dataSourceMode === "preprocessed" && (
                <>
                  <div className="space-y-2">
                    <Label className="text-sm">Chunking Method</Label>
                    <Select
                      value={config.chunkingMethod}
                      onValueChange={(v) =>
                        setConfig((prev) => ({ ...prev, chunkingMethod: v }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {CHUNKING_METHODS.map((method) => (
                          <SelectItem key={method.value} value={method.value}>
                            <div className="flex items-center gap-2">
                              <span>{method.label}</span>
                              <span className="text-xs text-muted-foreground">
                                ({method.description})
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    {/* Chunk Size */}
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label className="text-sm">Chunk Size</Label>
                        <span className="text-sm text-muted-foreground">
                          {config.chunkSize}
                        </span>
                      </div>
                      <Slider
                        value={[config.chunkSize]}
                        onValueChange={([v]) =>
                          setConfig((prev) => ({ ...prev, chunkSize: v }))
                        }
                        min={200}
                        max={4000}
                        step={100}
                      />
                    </div>

                    {/* Chunk Overlap */}
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label className="text-sm">Chunk Overlap</Label>
                        <span className="text-sm text-muted-foreground">
                          {config.chunkOverlap}
                        </span>
                      </div>
                      <Slider
                        value={[config.chunkOverlap]}
                        onValueChange={([v]) =>
                          setConfig((prev) => ({ ...prev, chunkOverlap: v }))
                        }
                        min={0}
                        max={config.chunkSize - 100}
                        step={50}
                      />
                    </div>
                  </div>
                </>
              )}
            </CollapsibleContent>
          </Collapsible>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleStart} disabled={isStartDisabled}>
            {isStarting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Evaluation
                {selectedDataset && (
                  <Badge variant="secondary" className="ml-2">
                    {selectedDataset.pair_count} pairs
                  </Badge>
                )}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

EvaluationConfigModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onOpenChange: PropTypes.func.isRequired,
  onStartEvaluation: PropTypes.func,
};

export default EvaluationConfigModal;
