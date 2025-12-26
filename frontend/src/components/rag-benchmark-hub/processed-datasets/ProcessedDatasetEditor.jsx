import { useCallback, useEffect, useState } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { Badge } from "../../ui/badge";
import { ScrollArea } from "../../ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Database,
  FileText,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
import { cn } from "../../../lib/utils";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function ProcessedDatasetEditor() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Chunk pagination
  const [chunks, setChunks] = useState([]);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalChunks, setTotalChunks] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const chunksPerPage = 20;

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingLog, setProcessingLog] = useState([]);

  const fetchDatasets = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/processed-datasets`);
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.status}`);
      }
      const data = await response.json();
      setDatasets(data.datasets || []);
    } catch (err) {
      console.error("Error fetching datasets:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchChunks = async (datasetId, page = 1, query = "") => {
    setIsLoadingChunks(true);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: chunksPerPage.toString(),
      });
      if (query) {
        params.append("search", query);
      }

      const response = await fetch(
        `${API_BASE_URL}/api/processed-datasets/${datasetId}/chunks?${params}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch chunks");
      }
      const data = await response.json();
      setChunks(data.chunks || []);
      setTotalChunks(data.total || 0);
    } catch (err) {
      console.error("Error fetching chunks:", err);
      setError(err.message);
    } finally {
      setIsLoadingChunks(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  useEffect(() => {
    if (selectedDataset) {
      fetchChunks(selectedDataset.id, currentPage, searchQuery);
    }
  }, [selectedDataset, currentPage, searchQuery]);

  const handleSelectDataset = (datasetId) => {
    const dataset = datasets.find((ds) => ds.id.toString() === datasetId);
    setSelectedDataset(dataset);
    setCurrentPage(1);
    setSearchQuery("");
    setChunks([]);
  };

  const handleDeleteDataset = async (datasetId) => {
    if (!confirm("Are you sure you want to delete this processed dataset?")) {
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/processed-datasets/${datasetId}`,
        { method: "DELETE" }
      );

      if (!response.ok) {
        throw new Error("Failed to delete dataset");
      }

      setDatasets((prev) => prev.filter((ds) => ds.id !== datasetId));
      if (selectedDataset?.id === datasetId) {
        setSelectedDataset(null);
        setChunks([]);
      }
    } catch (err) {
      console.error("Error deleting dataset:", err);
      setError(err.message);
    }
  };

  const handleStartProcessing = async (datasetId) => {
    setIsProcessing(true);
    setProcessingLog([]);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/processed-datasets/${datasetId}/process/stream`,
        { method: "POST" }
      );

      if (!response.ok) {
        throw new Error("Failed to start processing");
      }

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
              setProcessingLog((prev) => [...prev, data]);

              if (data.status === "completed" || data.status === "error") {
                setIsProcessing(false);
                await fetchDatasets();
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
      console.error("Processing error:", err);
      setError(err.message);
      setIsProcessing(false);
    }
  };

  const totalPages = Math.ceil(totalChunks / chunksPerPage);

  const getStatusColor = (status) => {
    switch (status) {
      case "completed":
        return "bg-green-500";
      case "processing":
        return "bg-blue-500";
      case "failed":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isLoading && datasets.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 h-[calc(100vh-200px)]">
      {/* Dataset List */}
      <Card className="flex flex-col">
        <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle>Processed Datasets</CardTitle>
            <CardDescription>Chunked and indexed datasets</CardDescription>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={fetchDatasets}
            disabled={isLoading}
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
        </CardHeader>
        <CardContent className="flex-1 p-0">
          {error && (
            <div className="mx-4 mb-2 flex items-center gap-2 text-destructive text-sm">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}
          <ScrollArea className="h-full px-4 pb-4">
            {datasets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Database className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No processed datasets yet</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Use the Preprocessing tab to create one
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className={cn(
                      "flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors",
                      selectedDataset?.id === dataset.id
                        ? "bg-accent"
                        : "hover:bg-muted"
                    )}
                    onClick={() => handleSelectDataset(dataset.id.toString())}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <Database className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium truncate">{dataset.name}</p>
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-xs",
                              getStatusColor(dataset.processing_status)
                            )}
                          >
                            {dataset.processing_status}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {dataset.chunk_count || 0} chunks ·{" "}
                          {dataset.vector_backend}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      {dataset.processing_status === "pending" && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartProcessing(dataset.id);
                          }}
                          disabled={isProcessing}
                        >
                          <Play className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteDataset(dataset.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Dataset Details / Chunks */}
      <Card className="flex flex-col">
        {selectedDataset ? (
          <>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>{selectedDataset.name}</CardTitle>
                  <CardDescription>
                    {selectedDataset.description || "No description"}
                  </CardDescription>
                </div>
                <Badge
                  variant="outline"
                  className={getStatusColor(selectedDataset.processing_status)}
                >
                  {selectedDataset.processing_status}
                </Badge>
              </div>
              <div className="flex flex-wrap gap-2 text-sm text-muted-foreground mt-2">
                <span>Backend: {selectedDataset.vector_backend}</span>
                <span>·</span>
                <span>
                  Chunking: {selectedDataset.preprocessing_config?.chunking?.method || "N/A"}
                </span>
                <span>·</span>
                <span>Created: {formatDate(selectedDataset.created_at)}</span>
              </div>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col gap-4">
              {/* Processing Log */}
              {isProcessing && (
                <div className="bg-muted rounded-md p-3 max-h-32 overflow-auto">
                  <p className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing...
                  </p>
                  {processingLog.map((log, idx) => (
                    <p key={idx} className="text-xs text-muted-foreground">
                      {log.step}: {log.message}
                    </p>
                  ))}
                </div>
              )}

              {/* Search */}
              {selectedDataset.processing_status === "completed" && (
                <>
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search chunks..."
                        value={searchQuery}
                        onChange={(e) => {
                          setSearchQuery(e.target.value);
                          setCurrentPage(1);
                        }}
                        className="pl-8"
                      />
                    </div>
                  </div>

                  {/* Chunks List */}
                  <ScrollArea className="flex-1">
                    {isLoadingChunks ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : chunks.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        {searchQuery
                          ? "No chunks match your search"
                          : "No chunks available"}
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {chunks.map((chunk, idx) => (
                          <div
                            key={chunk.id || idx}
                            className="p-3 rounded-md border bg-card hover:bg-muted/50 transition-colors"
                          >
                            <div className="flex items-start gap-2 mb-2">
                              <FileText className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                              <div className="flex-1 min-w-0">
                                <p className="text-xs text-muted-foreground">
                                  Source: {chunk.metadata?.source || "Unknown"} ·
                                  Chunk {(currentPage - 1) * chunksPerPage + idx + 1}
                                </p>
                              </div>
                            </div>
                            <p className="text-sm line-clamp-3">
                              {chunk.content || chunk.page_content}
                            </p>
                            {chunk.metadata?.summary && (
                              <p className="text-xs text-muted-foreground mt-2 italic">
                                Summary: {chunk.metadata.summary}
                              </p>
                            )}
                            {chunk.metadata?.keywords && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {chunk.metadata.keywords.slice(0, 5).map((kw, i) => (
                                  <Badge key={i} variant="secondary" className="text-xs">
                                    {kw}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </ScrollArea>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between pt-2 border-t">
                      <p className="text-sm text-muted-foreground">
                        {totalChunks} chunks total
                      </p>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                          disabled={currentPage === 1}
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <span className="text-sm">
                          Page {currentPage} of {totalPages}
                        </span>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                          disabled={currentPage === totalPages}
                        >
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Pending State */}
              {selectedDataset.processing_status === "pending" && !isProcessing && (
                <div className="flex-1 flex flex-col items-center justify-center text-center">
                  <Play className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-lg font-medium mb-2">Ready to Process</p>
                  <p className="text-muted-foreground mb-4">
                    This dataset hasn&apos;t been processed yet
                  </p>
                  <Button onClick={() => handleStartProcessing(selectedDataset.id)}>
                    <Play className="h-4 w-4 mr-2" />
                    Start Processing
                  </Button>
                </div>
              )}

              {/* Failed State */}
              {selectedDataset.processing_status === "failed" && (
                <div className="flex-1 flex flex-col items-center justify-center text-center">
                  <AlertCircle className="h-12 w-12 text-destructive mb-4" />
                  <p className="text-lg font-medium mb-2">Processing Failed</p>
                  <p className="text-muted-foreground mb-4">
                    {selectedDataset.error_message || "An error occurred during processing"}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => handleStartProcessing(selectedDataset.id)}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry Processing
                  </Button>
                </div>
              )}
            </CardContent>
          </>
        ) : (
          <CardContent className="flex-1 flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-4" />
              <p>Select a dataset to view chunks</p>
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
}

export default ProcessedDatasetEditor;
