import { useEffect, useState } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { ScrollArea } from "../../ui/scroll-area";
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
  AlertCircle,
  CheckCircle2,
  Loader2,
  Play,
} from "lucide-react";
import { cn } from "../../../lib/utils";
import { API_BASE_URL } from "../../../lib/api-config";

export function PreprocessingPipeline() {
  // Source dataset selection
  const [rawDatasets, setRawDatasets] = useState([]);
  const [selectedRawDatasetId, setSelectedRawDatasetId] = useState("");
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);

  // Output configuration
  const [outputName, setOutputName] = useState("");
  const [outputDescription, setOutputDescription] = useState("");

  // Cleaning configuration
  const [cleaningEnabled, setCleaningEnabled] = useState(false);
  const [removeHeaders, setRemoveHeaders] = useState(false);
  const [normalizeWhitespace, setNormalizeWhitespace] = useState(true);

  // LLM Metadata configuration
  const [metadataEnabled, setMetadataEnabled] = useState(false);
  const [extractSummary, setExtractSummary] = useState(true);
  const [extractKeywords, setExtractKeywords] = useState(true);
  const [extractEntities, setExtractEntities] = useState(true);
  const [extractCategories, setExtractCategories] = useState(true);

  // Chunking configuration
  const [chunkingMethod, setChunkingMethod] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);

  // Vector store configuration
  const [vectorBackend, setVectorBackend] = useState("pgvector");
  const [embedderModel, setEmbedderModel] = useState("all-MiniLM-L6-v2");

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRawDatasets();
  }, []);

  const fetchRawDatasets = async () => {
    setIsLoadingDatasets(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/raw-datasets`);
      if (!response.ok) throw new Error("Failed to fetch datasets");
      const data = await response.json();
      setRawDatasets(data.datasets || []);
    } catch (err) {
      console.error("Error fetching datasets:", err);
      setError(err.message);
    } finally {
      setIsLoadingDatasets(false);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedRawDatasetId || !outputName.trim()) {
      setError("Please select a source dataset and provide an output name");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProcessingStatus({ step: "Creating dataset...", progress: 0 });

    try {
      // Create the processed dataset
      const response = await fetch(
        `${API_BASE_URL}/api/processed-datasets?start_processing=true`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: outputName.trim(),
            description: outputDescription.trim() || null,
            raw_dataset_id: parseInt(selectedRawDatasetId),
            embedder_config: {
              model_name: embedderModel,
              model_type: "huggingface",
            },
            preprocessing_config: {
              cleaning: {
                enabled: cleaningEnabled,
                remove_headers_footers: removeHeaders,
                normalize_whitespace: normalizeWhitespace,
              },
              llm_metadata: {
                enabled: metadataEnabled,
                extract_summary: extractSummary,
                extract_keywords: extractKeywords,
                extract_entities: extractEntities,
                extract_categories: extractCategories,
              },
              chunking: {
                method: chunkingMethod,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
              },
            },
            vector_backend: vectorBackend,
          }),
        }
      );

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to create dataset");
      }

      const result = await response.json();
      setProcessingStatus({
        step: "Processing started",
        progress: 100,
        datasetId: result.id,
      });

      // Reset form
      setOutputName("");
      setOutputDescription("");
      setSelectedRawDatasetId("");
    } catch (err) {
      console.error("Processing error:", err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const selectedDataset = rawDatasets.find(
    (ds) => ds.id.toString() === selectedRawDatasetId
  );

  return (
    <ScrollArea className="h-[calc(100vh-200px)]">
      <div className="space-y-6 pr-4">
        {/* Source Dataset */}
        <Card>
          <CardHeader>
            <CardTitle>Source Dataset</CardTitle>
            <CardDescription>Select a raw dataset to process</CardDescription>
          </CardHeader>
          <CardContent>
            <Select
              value={selectedRawDatasetId}
              onValueChange={setSelectedRawDatasetId}
              disabled={isLoadingDatasets}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a raw dataset..." />
              </SelectTrigger>
              <SelectContent>
                {rawDatasets.map((ds) => (
                  <SelectItem key={ds.id} value={ds.id.toString()}>
                    {ds.name} ({ds.total_file_count} files)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedDataset && (
              <p className="text-sm text-muted-foreground mt-2">
                {selectedDataset.total_file_count} files,{" "}
                {(selectedDataset.total_size_bytes / 1024 / 1024).toFixed(1)} MB
              </p>
            )}
          </CardContent>
        </Card>

        {/* Cleaning (Optional) */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle>1. Cleaning (Optional)</CardTitle>
              <CardDescription>Clean and normalize text content</CardDescription>
            </div>
            <Switch checked={cleaningEnabled} onCheckedChange={setCleaningEnabled} />
          </CardHeader>
          {cleaningEnabled && (
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Remove headers/footers</Label>
                <Switch checked={removeHeaders} onCheckedChange={setRemoveHeaders} />
              </div>
              <div className="flex items-center justify-between">
                <Label>Normalize whitespace</Label>
                <Switch
                  checked={normalizeWhitespace}
                  onCheckedChange={setNormalizeWhitespace}
                />
              </div>
            </CardContent>
          )}
        </Card>

        {/* LLM Metadata (Optional) */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle>2. Metadata Extraction (Optional)</CardTitle>
              <CardDescription>Extract metadata using LLM</CardDescription>
            </div>
            <Switch checked={metadataEnabled} onCheckedChange={setMetadataEnabled} />
          </CardHeader>
          {metadataEnabled && (
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Extract summaries</Label>
                <Switch checked={extractSummary} onCheckedChange={setExtractSummary} />
              </div>
              <div className="flex items-center justify-between">
                <Label>Extract keywords</Label>
                <Switch checked={extractKeywords} onCheckedChange={setExtractKeywords} />
              </div>
              <div className="flex items-center justify-between">
                <Label>Extract entities</Label>
                <Switch checked={extractEntities} onCheckedChange={setExtractEntities} />
              </div>
              <div className="flex items-center justify-between">
                <Label>Extract categories</Label>
                <Switch
                  checked={extractCategories}
                  onCheckedChange={setExtractCategories}
                />
              </div>
            </CardContent>
          )}
        </Card>

        {/* Chunking (Required) */}
        <Card>
          <CardHeader>
            <CardTitle>3. Chunking (Required)</CardTitle>
            <CardDescription>Configure document chunking strategy</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label>Chunking Method</Label>
              <Select value={chunkingMethod} onValueChange={setChunkingMethod}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recursive">
                    Recursive - Split at natural boundaries
                  </SelectItem>
                  <SelectItem value="fixed">
                    Fixed Size - Equal chunk sizes
                  </SelectItem>
                  <SelectItem value="semantic">
                    Semantic - Based on meaning
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Chunk Size</Label>
                <span className="text-sm text-muted-foreground">
                  {chunkSize} characters
                </span>
              </div>
              <Slider
                value={[chunkSize]}
                onValueChange={([v]) => setChunkSize(v)}
                min={100}
                max={4000}
                step={100}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Chunk Overlap</Label>
                <span className="text-sm text-muted-foreground">
                  {chunkOverlap} characters
                </span>
              </div>
              <Slider
                value={[chunkOverlap]}
                onValueChange={([v]) => setChunkOverlap(v)}
                min={0}
                max={500}
                step={25}
              />
            </div>
          </CardContent>
        </Card>

        {/* Vector Database (Required) */}
        <Card>
          <CardHeader>
            <CardTitle>4. Vector Database (Required)</CardTitle>
            <CardDescription>Configure indexing settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Vector Backend</Label>
              <Select value={vectorBackend} onValueChange={setVectorBackend}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pgvector">PostgreSQL + pgvector</SelectItem>
                  <SelectItem value="qdrant">Qdrant</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Embedding Model</Label>
              <Select value={embedderModel} onValueChange={setEmbedderModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all-MiniLM-L6-v2">
                    all-MiniLM-L6-v2 (384 dim)
                  </SelectItem>
                  <SelectItem value="BAAI/bge-small-en-v1.5">
                    BGE-small-en (384 dim)
                  </SelectItem>
                  <SelectItem value="nomic-ai/nomic-embed-text-v1">
                    Nomic Embed (768 dim)
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Output Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Output Configuration</CardTitle>
            <CardDescription>Name your processed dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Dataset Name</Label>
              <Input
                placeholder="my-processed-dataset"
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Description (optional)</Label>
              <Input
                placeholder="Processed with semantic chunking..."
                value={outputDescription}
                onChange={(e) => setOutputDescription(e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 text-destructive p-4 bg-destructive/10 rounded-md">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        )}

        {/* Success Display */}
        {processingStatus?.progress === 100 && (
          <div className="flex items-center gap-2 text-green-600 p-4 bg-green-100 dark:bg-green-900/20 rounded-md">
            <CheckCircle2 className="h-5 w-5" />
            Processing started! Check the Processed Datasets tab for status.
          </div>
        )}

        {/* Start Processing Button */}
        <Button
          className="w-full"
          size="lg"
          onClick={handleStartProcessing}
          disabled={isProcessing || !selectedRawDatasetId || !outputName.trim()}
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Play className="h-5 w-5 mr-2" />
              Start Processing
            </>
          )}
        </Button>
      </div>
    </ScrollArea>
  );
}

export default PreprocessingPipeline;
