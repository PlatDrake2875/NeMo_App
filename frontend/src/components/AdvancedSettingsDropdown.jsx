import PropTypes from "prop-types";
import { useEffect, useState } from "react";
import { Database, Zap, Search, Loader2, RefreshCw } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Separator } from "./ui/separator";
import { getApiBaseUrl } from "../lib/api-config";

/**
 * AdvancedSettingsDropdown - Dialog for configuring RAG settings
 * Allows users to select dataset, toggle RAG and ColBERT
 */
export function AdvancedSettingsDropdown({
  open,
  onOpenChange,
  selectedDataset,
  onDatasetChange,
  isRagEnabled = true,
  onRagEnabledChange,
  isColbertEnabled = true,
  onColbertEnabledChange,
}) {
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch datasets when dialog opens
  useEffect(() => {
    if (open) {
      fetchDatasets();
    }
  }, [open]);

  const fetchDatasets = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/processed-datasets`);
      if (!response.ok) {
        throw new Error("Failed to fetch datasets");
      }
      const data = await response.json();

      // Build list: processed datasets + default rag_documents
      const datasets = [];

      // Add default collection first
      datasets.push({
        name: "rag_documents",
        collection_name: "rag_documents",
        chunk_count: null,
        isDefault: true,
      });

      // Add processed datasets (filter to only completed ones with chunks indexed)
      // Note: Only datasets with chunk_count > 0 can be used for RAG chat
      // New-flow datasets (document_count > 0 but chunk_count = 0) need to be
      // evaluated first to create embeddings before they can be used for chat
      if (data.datasets && Array.isArray(data.datasets)) {
        data.datasets
          .filter(d => d.processing_status === "completed" && d.chunk_count > 0)
          .forEach(d => {
            // Don't duplicate if someone named their dataset "rag_documents"
            if (d.collection_name !== "rag_documents") {
              datasets.push({
                name: d.name,
                collection_name: d.collection_name,
                chunk_count: d.chunk_count,
                isDefault: false,
              });
            }
          });
      }

      setAvailableDatasets(datasets);

      // If no dataset selected yet, select the default
      if (!selectedDataset && datasets.length > 0) {
        onDatasetChange?.(datasets[0].collection_name);
      }
    } catch (err) {
      console.error("Error fetching datasets:", err);
      setError(err.message);
      // Still show default on error
      setAvailableDatasets([{
        name: "rag_documents",
        collection_name: "rag_documents",
        chunk_count: null,
        isDefault: true,
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Advanced Settings
          </DialogTitle>
          <DialogDescription>
            Configure RAG retrieval settings for your conversation.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-6">
          {/* Dataset Selection */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="dataset-select" className="text-sm font-medium">
                Dataset / Collection
              </Label>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={fetchDatasets}
                disabled={isLoading}
                title="Refresh datasets"
              >
                <RefreshCw className={`h-3 w-3 ${isLoading ? "animate-spin" : ""}`} />
              </Button>
            </div>
            <Select
              value={selectedDataset || "rag_documents"}
              onValueChange={onDatasetChange}
              disabled={isLoading}
            >
              <SelectTrigger id="dataset-select" className="w-full">
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Loading...</span>
                  </div>
                ) : (
                  <SelectValue placeholder="Select a dataset..." />
                )}
              </SelectTrigger>
              <SelectContent>
                {availableDatasets.map((dataset) => (
                  <SelectItem key={dataset.collection_name} value={dataset.collection_name}>
                    <div className="flex items-center gap-2">
                      <span>{dataset.name}</span>
                      {dataset.isDefault && (
                        <span className="text-xs text-muted-foreground">(default)</span>
                      )}
                      {dataset.chunk_count !== null && (
                        <span className="text-xs text-muted-foreground">
                          ({dataset.chunk_count} chunks)
                        </span>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {error && (
              <p className="text-xs text-destructive">{error}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Select which document collection to search during RAG retrieval.
            </p>
          </div>

          <Separator />

          {/* RAG Toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label
                htmlFor="rag-toggle"
                className="text-sm font-medium flex items-center gap-2"
              >
                <Search className="h-4 w-4" />
                Enable RAG
              </Label>
              <p className="text-xs text-muted-foreground">
                Retrieve relevant documents to augment responses
              </p>
            </div>
            <Switch
              id="rag-toggle"
              checked={isRagEnabled}
              onCheckedChange={onRagEnabledChange}
            />
          </div>

          {/* ColBERT Toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label
                htmlFor="colbert-toggle"
                className="text-sm font-medium flex items-center gap-2"
              >
                <Zap className="h-4 w-4" />
                Enable ColBERT Reranking
              </Label>
              <p className="text-xs text-muted-foreground">
                Use two-stage retrieval with ColBERT for better accuracy
              </p>
            </div>
            <Switch
              id="colbert-toggle"
              checked={isColbertEnabled}
              onCheckedChange={onColbertEnabledChange}
              disabled={!isRagEnabled}
            />
          </div>

          {!isRagEnabled && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              RAG is disabled. The model will respond without document retrieval.
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

AdvancedSettingsDropdown.propTypes = {
  open: PropTypes.bool.isRequired,
  onOpenChange: PropTypes.func.isRequired,
  selectedDataset: PropTypes.string,
  onDatasetChange: PropTypes.func,
  isRagEnabled: PropTypes.bool,
  onRagEnabledChange: PropTypes.func,
  isColbertEnabled: PropTypes.bool,
  onColbertEnabledChange: PropTypes.func,
};
