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
import { Input } from "../../ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import { Badge } from "../../ui/badge";
import {
  Loader2,
  RefreshCw,
  Sparkles,
  Trash2,
  Database,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";

export function GenerateQAPage() {
  // Available collections for Q&A generation
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState("");

  // Q&A Generation state
  const [evalDatasets, setEvalDatasets] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationConfig, setGenerationConfig] = useState({
    name: "",
    pairsPerChunk: 2,
    maxChunks: 50,
  });
  const [generationResult, setGenerationResult] = useState(null);

  // Fetch collections and eval datasets on mount
  useEffect(() => {
    fetchCollections();
    fetchEvalDatasets();
  }, []);

  const fetchCollections = async () => {
    setCollectionsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/processed-datasets`);
      if (response.ok) {
        const data = await response.json();
        const datasets = data.datasets || data;
        const completedDatasets = datasets.filter((d) => d.processing_status === "completed");
        setCollections(completedDatasets);
        // Set default collection if available
        if (completedDatasets.length > 0 && !selectedCollection) {
          setSelectedCollection(completedDatasets[0].collection_name);
        }
      }
    } catch (err) {
      console.error("Error fetching collections:", err);
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
      console.error("Error fetching eval datasets:", err);
    }
  };

  // Get the selected collection's full info
  const getSelectedCollection = () => {
    if (!selectedCollection || selectedCollection === "rag_documents") return null;
    return collections.find(c => c.collection_name === selectedCollection);
  };

  // Generate Q&A dataset
  const generateQADataset = async () => {
    const selectedCol = getSelectedCollection();
    if (!selectedCol) {
      alert("Please select a document collection first");
      return;
    }

    if (!generationConfig.name.trim()) {
      alert("Please enter a name for the evaluation dataset");
      return;
    }

    setIsGenerating(true);
    setGenerationResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          processed_dataset_id: selectedCol.id,
          name: generationConfig.name.trim(),
          model: "openai/gpt-4o-mini",
          pairs_per_chunk: generationConfig.pairsPerChunk,
          max_chunks: generationConfig.maxChunks,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setGenerationResult(result);
        // Refresh eval datasets
        await fetchEvalDatasets();
        setGenerationConfig(prev => ({ ...prev, name: "" }));
      } else {
        const error = await response.json();
        alert(`Generation failed: ${error.detail || error.message || JSON.stringify(error)}`);
      }
    } catch (err) {
      console.error("Error generating Q&A dataset:", err);
      alert(`Failed to generate: ${err.message}`);
    } finally {
      setIsGenerating(false);
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
      }
    } catch (err) {
      console.error("Error deleting eval dataset:", err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Sparkles className="h-6 w-6" />
            Generate Q&A Dataset
          </h2>
          <p className="text-muted-foreground">
            Create Q&A pairs from your documents using AI for evaluation
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={fetchCollections}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="max-w-2xl mx-auto space-y-6">
        {/* Source Collection Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="h-5 w-5" />
              Source Collection
            </CardTitle>
            <CardDescription>
              Select the document collection to generate Q&A pairs from
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Document Collection</Label>
              <Select
                value={selectedCollection}
                onValueChange={setSelectedCollection}
                disabled={collectionsLoading}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select collection..." />
                </SelectTrigger>
                <SelectContent>
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
              {selectedCollection && selectedCollection !== "rag_documents" && (
                <p className="text-xs text-muted-foreground">
                  Embedder: {
                    collections.find(c => c.collection_name === selectedCollection)?.embedder_config?.model_name?.split("/").pop() || "unknown"
                  }
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Q&A Dataset Generation Card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              Generate Evaluation Dataset
            </CardTitle>
            <CardDescription>
              Create Q&A pairs from your documents using AI (GPT-4o-mini)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Dataset Name */}
            <div className="space-y-2">
              <Label>Dataset Name</Label>
              <Input
                placeholder="e.g., AI-Video-Detection-QA"
                value={generationConfig.name}
                onChange={(e) =>
                  setGenerationConfig((prev) => ({ ...prev, name: e.target.value }))
                }
                disabled={isGenerating}
              />
            </div>

            {/* Generation Settings */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Pairs per Chunk</Label>
                <Select
                  value={String(generationConfig.pairsPerChunk)}
                  onValueChange={(value) =>
                    setGenerationConfig((prev) => ({ ...prev, pairsPerChunk: parseInt(value) }))
                  }
                  disabled={isGenerating}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1 pair</SelectItem>
                    <SelectItem value="2">2 pairs</SelectItem>
                    <SelectItem value="3">3 pairs</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Max Chunks</Label>
                <Select
                  value={String(generationConfig.maxChunks)}
                  onValueChange={(value) =>
                    setGenerationConfig((prev) => ({ ...prev, maxChunks: parseInt(value) }))
                  }
                  disabled={isGenerating}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10 chunks (~20 pairs)</SelectItem>
                    <SelectItem value="25">25 chunks (~50 pairs)</SelectItem>
                    <SelectItem value="50">50 chunks (~100 pairs)</SelectItem>
                    <SelectItem value="100">100 chunks (~200 pairs)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              Estimated: ~{generationConfig.pairsPerChunk * generationConfig.maxChunks} Q&A pairs
              {getSelectedCollection() && ` from "${getSelectedCollection()?.name}"`}
            </p>

            <Button
              className="w-full"
              variant="secondary"
              onClick={generateQADataset}
              disabled={isGenerating || !selectedCollection || !generationConfig.name.trim()}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating Q&A Pairs...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Generate Q&A Dataset
                </>
              )}
            </Button>

            {/* Generation Result */}
            {generationResult && (
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-md">
                <p className="text-sm text-green-600 dark:text-green-500">
                  Generated "{generationResult.name}" with {generationResult.pair_count} Q&A pairs
                  from {generationResult.chunks_processed} chunks
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Existing Evaluation Datasets */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="h-5 w-5" />
              Existing Evaluation Datasets
            </CardTitle>
            <CardDescription>
              Manage your generated Q&A datasets
            </CardDescription>
          </CardHeader>
          <CardContent>
            {evalDatasets.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No evaluation datasets yet. Generate one above.
              </p>
            ) : (
              <div className="space-y-2">
                {evalDatasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{dataset.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {dataset.pair_count} pairs
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Created: {new Date(dataset.created_at).toLocaleString()}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteEvalDataset(dataset.id)}
                      className="text-destructive hover:text-destructive ml-2"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default GenerateQAPage;
