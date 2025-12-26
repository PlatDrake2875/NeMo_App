import { useState } from "react";
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
  Database,
  Download,
  ExternalLink,
  Loader2,
  Search,
} from "lucide-react";
import { cn } from "../../../lib/utils";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function HuggingFaceImporter() {
  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Selected dataset
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetMetadata, setDatasetMetadata] = useState(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);

  // Import configuration
  const [importMode, setImportMode] = useState("raw"); // "raw" or "direct"
  const [datasetName, setDatasetName] = useState("");
  const [textColumn, setTextColumn] = useState("");
  const [split, setSplit] = useState("train");
  const [maxRows, setMaxRows] = useState(1000);

  // Direct processing options
  const [chunkingMethod, setChunkingMethod] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [vectorBackend, setVectorBackend] = useState("pgvector");
  const [embedderModel, setEmbedderModel] = useState("all-MiniLM-L6-v2");
  const [enableMetadataExtraction, setEnableMetadataExtraction] = useState(false);

  // Import state
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState([]);
  const [importComplete, setImportComplete] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setError(null);
    setSelectedDataset(null);
    setDatasetMetadata(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/huggingface/datasets/search?query=${encodeURIComponent(searchQuery)}`
      );

      if (!response.ok) {
        throw new Error("Search failed");
      }

      const data = await response.json();
      setSearchResults(data.datasets || []);

      // Check for server-side search error
      if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      console.error("Search error:", err);
      setError(err.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectDataset = async (dataset) => {
    setSelectedDataset(dataset);
    setDatasetName(dataset.id.replace("/", "_"));
    setIsLoadingMetadata(true);
    setImportComplete(false);
    setImportProgress([]);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/huggingface/datasets/${encodeURIComponent(dataset.id)}/metadata`
      );

      if (!response.ok) {
        throw new Error("Failed to load dataset metadata");
      }

      const metadata = await response.json();
      setDatasetMetadata(metadata);

      // Auto-select text column if available
      if (metadata.columns?.length > 0) {
        const textCol = metadata.columns.find((col) =>
          ["text", "content", "question", "answer", "document"].includes(col.name.toLowerCase())
        );
        if (textCol) {
          setTextColumn(textCol.name);
        } else {
          setTextColumn(metadata.columns[0].name);
        }
      }

      // Auto-select split
      if (metadata.splits?.length > 0) {
        setSplit(metadata.splits[0]);
      }
    } catch (err) {
      console.error("Metadata error:", err);
      setError(err.message);
    } finally {
      setIsLoadingMetadata(false);
    }
  };

  const handleImport = async () => {
    if (!selectedDataset || !datasetName.trim() || !textColumn) {
      setError("Please fill in all required fields");
      return;
    }

    setIsImporting(true);
    setImportProgress([]);
    setImportComplete(false);
    setError(null);

    const endpoint =
      importMode === "raw"
        ? `${API_BASE_URL}/api/huggingface/import-raw`
        : `${API_BASE_URL}/api/huggingface/process-direct`;

    // Build request body matching backend schema
    const hfConfig = {
      dataset_id: selectedDataset.id,
      text_column: textColumn,
      split: split,
      max_samples: maxRows,
    };

    const body =
      importMode === "raw"
        ? {
            hf_config: hfConfig,
            raw_dataset_name: datasetName.trim(),
            description: null,
          }
        : {
            hf_config: hfConfig,
            processed_dataset_name: datasetName.trim(),
            description: null,
            preprocessing_config: {
              cleaning: { enabled: false },
              llm_metadata: {
                enabled: enableMetadataExtraction,
                extract_summary: true,
                extract_keywords: true,
                extract_entities: true,
                extract_categories: true,
              },
              chunking: {
                method: chunkingMethod,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
              },
            },
            vector_backend: vectorBackend,
            embedder_config: {
              model_name: embedderModel,
              model_type: "huggingface",
            },
          };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Import failed");
      }

      // Stream SSE response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              setImportProgress((prev) => [...prev, data]);

              // Backend sends "type" field: "status", "progress", "completed", "error"
              if (data.type === "completed") {
                setImportComplete(true);
                setIsImporting(false);
              } else if (data.type === "error") {
                setError(data.message || "Import failed");
                setIsImporting(false);
              }
            } catch (parseError) {
              // Log if it looks like JSON but failed to parse
              if (line.slice(6).trim().startsWith('{')) {
                console.warn('Failed to parse SSE JSON:', line, parseError);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error("Import error:", err);
      setError(err.message);
      setIsImporting(false);
    }
  };

  return (
    <ScrollArea className="h-[calc(100vh-200px)]">
      <div className="space-y-6 pr-4">
        {/* Search Section */}
        <Card>
          <CardHeader>
            <CardTitle>Search HuggingFace Datasets</CardTitle>
            <CardDescription>
              Find and import datasets from the HuggingFace Hub
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search datasets (e.g., microsoft/wiki_qa)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  className="pl-8"
                />
              </div>
              <Button onClick={handleSearch} disabled={isSearching}>
                {isSearching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "Search"
                )}
              </Button>
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="mt-4 space-y-2">
                {searchResults.map((dataset) => (
                  <div
                    key={dataset.id}
                    className={cn(
                      "flex items-center justify-between p-3 rounded-md border cursor-pointer transition-colors",
                      selectedDataset?.id === dataset.id
                        ? "bg-accent border-primary"
                        : "hover:bg-muted"
                    )}
                    onClick={() => handleSelectDataset(dataset)}
                  >
                    <div className="flex items-center gap-3">
                      <Database className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="font-medium">{dataset.id}</p>
                        <p className="text-xs text-muted-foreground">
                          {dataset.downloads?.toLocaleString() || 0} downloads ·{" "}
                          {dataset.likes || 0} likes
                        </p>
                      </div>
                    </div>
                    <a
                      href={`https://huggingface.co/datasets/${dataset.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-muted-foreground hover:text-primary"
                    >
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dataset Configuration */}
        {selectedDataset && (
          <Card>
            <CardHeader>
              <CardTitle>Configure Import</CardTitle>
              <CardDescription>
                {isLoadingMetadata
                  ? "Loading dataset information..."
                  : `Import ${selectedDataset.id}`}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {isLoadingMetadata ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <>
                  {/* Import Mode */}
                  <div className="space-y-2">
                    <Label>Import Mode</Label>
                    <Select value={importMode} onValueChange={setImportMode}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="raw">
                          Import as Raw Dataset (process later)
                        </SelectItem>
                        <SelectItem value="direct">
                          Direct Processing (chunk and index now)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Dataset Name */}
                  <div className="space-y-2">
                    <Label>Dataset Name</Label>
                    <Input
                      value={datasetName}
                      onChange={(e) => setDatasetName(e.target.value)}
                      placeholder="my-dataset"
                    />
                  </div>

                  {/* Column & Split Selection */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Text Column</Label>
                      <Select value={textColumn} onValueChange={setTextColumn}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select column..." />
                        </SelectTrigger>
                        <SelectContent>
                          {datasetMetadata?.columns?.map((col) => (
                            <SelectItem key={col.name} value={col.name}>
                              {col.name} ({col.dtype})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Split</Label>
                      <Select value={split} onValueChange={setSplit}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {datasetMetadata?.splits?.map((s) => (
                            <SelectItem key={s} value={s}>
                              {s}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Max Rows */}
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Max Rows</Label>
                      <span className="text-sm text-muted-foreground">
                        {maxRows.toLocaleString()} rows
                      </span>
                    </div>
                    <Slider
                      value={[maxRows]}
                      onValueChange={([v]) => setMaxRows(v)}
                      min={100}
                      max={10000}
                      step={100}
                    />
                  </div>

                  {/* Direct Processing Options */}
                  {importMode === "direct" && (
                    <>
                      <div className="border-t pt-4">
                        <h4 className="font-medium mb-4">Processing Options</h4>

                        {/* LLM Metadata */}
                        <div className="flex items-center justify-between mb-4">
                          <Label>Extract LLM Metadata</Label>
                          <Switch
                            checked={enableMetadataExtraction}
                            onCheckedChange={setEnableMetadataExtraction}
                          />
                        </div>

                        {/* Chunking */}
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label>Chunking Method</Label>
                            <Select
                              value={chunkingMethod}
                              onValueChange={setChunkingMethod}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="recursive">Recursive</SelectItem>
                                <SelectItem value="fixed">Fixed Size</SelectItem>
                                <SelectItem value="semantic">Semantic</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <Label>Chunk Size</Label>
                              <span className="text-sm text-muted-foreground">
                                {chunkSize} chars
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
                                {chunkOverlap} chars
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
                        </div>

                        {/* Vector Store */}
                        <div className="grid grid-cols-2 gap-4 mt-4">
                          <div className="space-y-2">
                            <Label>Vector Backend</Label>
                            <Select
                              value={vectorBackend}
                              onValueChange={setVectorBackend}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="pgvector">
                                  PostgreSQL + pgvector
                                </SelectItem>
                                <SelectItem value="qdrant">Qdrant</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label>Embedding Model</Label>
                            <Select
                              value={embedderModel}
                              onValueChange={setEmbedderModel}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="all-MiniLM-L6-v2">
                                  MiniLM-L6 (384d)
                                </SelectItem>
                                <SelectItem value="BAAI/bge-small-en-v1.5">
                                  BGE-small (384d)
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 text-destructive p-4 bg-destructive/10 rounded-md">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        )}

        {/* Import Progress */}
        {(isImporting || importProgress.length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {isImporting ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Importing...
                  </>
                ) : importComplete ? (
                  <>
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                    Import Complete
                  </>
                ) : (
                  "Import Progress"
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-48 overflow-auto">
                {importProgress.map((log, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "text-sm p-2 rounded",
                      log.type === "error"
                        ? "bg-destructive/10 text-destructive"
                        : log.type === "completed"
                        ? "bg-green-100 dark:bg-green-900/20 text-green-600"
                        : "bg-muted"
                    )}
                  >
                    {log.type && <span className="font-medium">[{log.type}] </span>}
                    {log.message}
                    {log.current !== undefined && log.total !== undefined && (
                      <span className="text-muted-foreground ml-2">
                        ({log.current}/{log.total})
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Import Button */}
        {selectedDataset && !isLoadingMetadata && (
          <Button
            className="w-full"
            size="lg"
            onClick={handleImport}
            disabled={isImporting || !datasetName.trim() || !textColumn}
          >
            {isImporting ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Download className="h-5 w-5 mr-2" />
                {importMode === "raw" ? "Import as Raw Dataset" : "Import & Process"}
              </>
            )}
          </Button>
        )}
      </div>
    </ScrollArea>
  );
}

export default HuggingFaceImporter;
