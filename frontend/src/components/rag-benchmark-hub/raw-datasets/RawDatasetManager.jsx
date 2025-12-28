import { useCallback, useEffect, useState } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { ScrollArea } from "../../ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../../ui/alert-dialog";
import {
  AlertCircle,
  Eye,
  FileText,
  FolderOpen,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { cn } from "../../../lib/utils";
import { API_BASE_URL } from "../../../lib/api-config";
import { DocumentPreviewModal } from "../shared/DocumentPreviewModal";

export function RawDatasetManager() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Create dialog state
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetDescription, setNewDatasetDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Preview state
  const [previewFile, setPreviewFile] = useState(null);

  // Delete confirmation state
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchDatasets = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/raw-datasets`);
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

  const fetchDatasetDetails = async (datasetId) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/raw-datasets/${datasetId}?include_files=true`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch dataset: ${response.status}`);
      }
      const data = await response.json();
      setSelectedDataset(data);
    } catch (err) {
      console.error("Error fetching dataset details:", err);
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const handleCreateDataset = async () => {
    if (!newDatasetName.trim()) return;

    setIsCreating(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/raw-datasets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newDatasetName.trim(),
          description: newDatasetDescription.trim() || null,
          source_type: "upload",
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to create dataset");
      }

      const newDataset = await response.json();
      setDatasets((prev) => [newDataset, ...prev]);
      setSelectedDataset(newDataset);
      setIsCreateDialogOpen(false);
      setNewDatasetName("");
      setNewDatasetDescription("");
    } catch (err) {
      console.error("Error creating dataset:", err);
      setError(err.message);
    } finally {
      setIsCreating(false);
    }
  };

  // Initiate delete - opens confirmation dialog
  const handleDeleteDataset = (datasetId) => {
    setPendingDeleteId(datasetId);
    setDeleteConfirmOpen(true);
  };

  // Confirm delete - actually deletes the dataset
  const confirmDeleteDataset = async () => {
    if (!pendingDeleteId) return;

    setIsDeleting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/raw-datasets/${pendingDeleteId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete dataset");
      }

      setDatasets((prev) => prev.filter((ds) => ds.id !== pendingDeleteId));
      if (selectedDataset?.id === pendingDeleteId) {
        setSelectedDataset(null);
      }
    } catch (err) {
      console.error("Error deleting dataset:", err);
      setError(err.message);
    } finally {
      setIsDeleting(false);
      setDeleteConfirmOpen(false);
      setPendingDeleteId(null);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files?.length || !selectedDataset) return;

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/raw-datasets/${selectedDataset.id}/files/batch`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Upload failed");
      }

      // Refresh dataset details
      await fetchDatasetDetails(selectedDataset.id);
      await fetchDatasets();
    } catch (err) {
      console.error("Error uploading files:", err);
      setError(err.message);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      event.target.value = "";
    }
  };

  const handleDeleteFile = async (fileId) => {
    if (!selectedDataset) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/raw-datasets/${selectedDataset.id}/files/${fileId}`,
        { method: "DELETE" }
      );

      if (!response.ok) {
        throw new Error("Failed to delete file");
      }

      // Refresh dataset details
      await fetchDatasetDetails(selectedDataset.id);
      await fetchDatasets();
    } catch (err) {
      console.error("Error deleting file:", err);
      setError(err.message);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
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
            <CardTitle>Raw Datasets</CardTitle>
            <CardDescription>Unprocessed document collections</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchDatasets}
              disabled={isLoading}
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>
            <Button size="sm" onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-1" />
              New
            </Button>
          </div>
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
                <FolderOpen className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No datasets yet</p>
                <Button
                  className="mt-4"
                  onClick={() => setIsCreateDialogOpen(true)}
                >
                  Create First Dataset
                </Button>
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
                    onClick={() => fetchDatasetDetails(dataset.id)}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <FolderOpen className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                      <div className="min-w-0">
                        <p className="font-medium truncate">{dataset.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {dataset.total_file_count} files ·{" "}
                          {formatBytes(dataset.total_size_bytes)}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="flex-shrink-0 opacity-0 group-hover:opacity-100 hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteDataset(dataset.id);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Dataset Detail */}
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
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDeleteDataset(selectedDataset.id)}
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              </div>
              <div className="flex gap-4 text-sm text-muted-foreground mt-2">
                <span>{selectedDataset.total_file_count} files</span>
                <span>{formatBytes(selectedDataset.total_size_bytes)}</span>
                <span>Created {formatDate(selectedDataset.created_at)}</span>
              </div>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col gap-4">
              {/* Upload Area */}
              <div className="border-2 border-dashed rounded-md p-4">
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  multiple
                  accept=".pdf,.json,.md,.txt,.csv"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
                <label
                  htmlFor="file-upload"
                  className="flex flex-col items-center cursor-pointer"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-2" />
                      <p className="text-sm text-muted-foreground">Uploading...</p>
                    </>
                  ) : (
                    <>
                      <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                      <p className="text-sm font-medium">
                        Click to upload files
                      </p>
                      <p className="text-xs text-muted-foreground">
                        PDF, JSON, Markdown, Text, CSV
                      </p>
                    </>
                  )}
                </label>
              </div>

              {/* Files List */}
              <ScrollArea className="flex-1">
                {selectedDataset.files?.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No files uploaded yet
                  </div>
                ) : (
                  <div className="space-y-2">
                    {selectedDataset.files?.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-muted group"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          <div className="min-w-0">
                            <p className="text-sm truncate">{file.filename}</p>
                            <p className="text-xs text-muted-foreground">
                              {file.file_type.toUpperCase()} ·{" "}
                              {formatBytes(file.size_bytes)}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() =>
                              setPreviewFile({
                                ...file,
                                datasetId: selectedDataset.id,
                              })
                            }
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 opacity-0 group-hover:opacity-100 hover:text-destructive"
                            onClick={() => handleDeleteFile(file.id)}
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
          </>
        ) : (
          <CardContent className="flex-1 flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <FolderOpen className="h-12 w-12 mx-auto mb-4" />
              <p>Select a dataset to view details</p>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Create Dataset Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Raw Dataset</DialogTitle>
            <DialogDescription>
              Create a new container for raw documents
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="my-dataset"
                value={newDatasetName}
                onChange={(e) => setNewDatasetName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Description (optional)</label>
              <Input
                placeholder="A collection of research papers"
                value={newDatasetDescription}
                onChange={(e) => setNewDatasetDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateDataset}
              disabled={!newDatasetName.trim() || isCreating}
            >
              {isCreating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* File Preview Modal */}
      {previewFile && (
        <DocumentPreviewModal
          file={previewFile}
          onClose={() => setPreviewFile(null)}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this dataset and all its files?
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel
              onClick={() => {
                setDeleteConfirmOpen(false);
                setPendingDeleteId(null);
              }}
              disabled={isDeleting}
            >
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteDataset}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default RawDatasetManager;
