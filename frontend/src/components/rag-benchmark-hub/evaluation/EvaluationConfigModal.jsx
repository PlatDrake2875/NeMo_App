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
import { Badge } from "../../ui/badge";
import { Separator } from "../../ui/separator";
import {
  Play,
  Loader2,
  Database,
  FlaskConical,
  AlertCircle,
  Settings2,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";

// Available embedding models
const AVAILABLE_EMBEDDERS = [
  { value: "sentence-transformers/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2", description: "Default, Fast (384d)" },
  { value: "sentence-transformers/all-mpnet-base-v2", label: "all-mpnet-base-v2", description: "Higher accuracy (768d)" },
  { value: "BAAI/bge-small-en-v1.5", label: "bge-small-en-v1.5", description: "BGE Small (384d)" },
  { value: "BAAI/bge-base-en-v1.5", label: "bge-base-en-v1.5", description: "BGE Base (768d)" },
  { value: "nomic-ai/nomic-embed-text-v1", label: "nomic-embed-text-v1", description: "Nomic, Long context (768d)" },
];

export function EvaluationConfigModal({ open, onOpenChange, onStartEvaluation }) {
  // Collections state
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);

  // Evaluation datasets state
  const [evalDatasets, setEvalDatasets] = useState([]);

  // Configuration state
  const [config, setConfig] = useState({
    collection: "",
    enableRag: true,
    embedder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "colbert",
    enableReranking: true,
    topK: 5,
    temperature: 0.1,
  });

  const [selectedEvalDataset, setSelectedEvalDataset] = useState("none");
  const [experimentName, setExperimentName] = useState("");
  const [isStarting, setIsStarting] = useState(false);

  // Fetch collections and datasets when modal opens, reset form
  useEffect(() => {
    if (open) {
      fetchCollections();
      fetchEvalDatasets();
      // Reset experiment name when modal opens
      setExperimentName("");
    }
  }, [open]);

  const fetchCollections = async () => {
    setCollectionsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/processed-datasets`);
      if (response.ok) {
        const data = await response.json();
        const datasets = data.datasets || [];
        const completed = datasets.filter(d => d.processing_status === "completed");
        setCollections(completed);
        if (completed.length > 0 && !config.collection) {
          setConfig(prev => ({ ...prev, collection: completed[0].collection_name }));
        }
      }
    } catch (err) {
      console.error("Failed to fetch collections:", err);
    } finally {
      setCollectionsLoading(false);
    }
  };

  const fetchEvalDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`);
      if (response.ok) {
        const data = await response.json();
        setEvalDatasets(data);
      }
    } catch (err) {
      console.error("Failed to fetch evaluation datasets:", err);
    }
  };

  const getSelectedCollection = () => {
    return collections.find(c => c.collection_name === config.collection);
  };

  const handleStart = async () => {
    if (!config.collection) return;

    setIsStarting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/tasks/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_name: experimentName.trim() || null,
          eval_dataset_id: selectedEvalDataset === "none" ? null : selectedEvalDataset,
          collection_name: config.collection,
          use_rag: config.enableRag,
          embedder: config.embedder,
          use_colbert: config.reranker === "colbert" && config.enableReranking,
          top_k: config.topK,
          temperature: config.temperature,
        }),
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
  const selectedDataset = evalDatasets.find(d => d.id === selectedEvalDataset);

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

          {/* Dataset Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4" />
              Evaluation Dataset
            </Label>
            <Select value={selectedEvalDataset} onValueChange={setSelectedEvalDataset}>
              <SelectTrigger>
                <SelectValue placeholder="Select dataset..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">
                  <div className="flex items-center gap-2">
                    <span>Quick Test (1 question)</span>
                    <Badge variant="secondary" className="text-xs">Fast</Badge>
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
                {selectedDataset.pair_count} Q&A pairs from {selectedDataset.source_dataset_id || "unknown source"}
              </p>
            )}
          </div>

          <Separator />

          {/* Collection Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Document Collection</Label>
            <Select
              value={config.collection}
              onValueChange={(v) => setConfig(prev => ({ ...prev, collection: v }))}
              disabled={collectionsLoading}
            >
              <SelectTrigger>
                <SelectValue placeholder={collectionsLoading ? "Loading..." : "Select collection..."} />
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
                  onCheckedChange={(v) => setConfig(prev => ({ ...prev, enableRag: v }))}
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
                  onCheckedChange={(v) => setConfig(prev => ({ ...prev, enableReranking: v }))}
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
                onValueChange={([v]) => setConfig(prev => ({ ...prev, topK: v }))}
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
                <span className="text-sm text-muted-foreground">{config.temperature.toFixed(2)}</span>
              </div>
              <Slider
                value={[config.temperature]}
                onValueChange={([v]) => setConfig(prev => ({ ...prev, temperature: v }))}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            {/* Embedder Selection */}
            <div className="space-y-2">
              <Label className="text-sm">Embedder Model</Label>
              <Select
                value={config.embedder}
                onValueChange={(v) => setConfig(prev => ({ ...prev, embedder: v }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {AVAILABLE_EMBEDDERS.map((emb) => (
                    <SelectItem key={emb.value} value={emb.value}>
                      <div className="flex items-center gap-2">
                        <span>{emb.label}</span>
                        <span className="text-xs text-muted-foreground">({emb.description})</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleStart} disabled={isStarting || !config.collection}>
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
