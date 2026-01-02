import { useCallback } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "./ui/button";
import { Toaster, toast } from "sonner";
import {
  ChunkingConfigCard,
  VectorStoreStatusCard,
  DocumentUploadCard,
  DocumentSourceList,
  DocumentStatsBar
} from "./document-manager";
import { useChunkingConfig } from "../hooks/useChunkingConfig";
import { useVectorStoreConfig } from "../hooks/useVectorStoreConfig";
import { useDocumentManager } from "../hooks/useDocumentManager";

export function DocumentManager({ onBack }) {
  // Chunking configuration
  const {
    method: chunkingMethod,
    setMethod: setChunkingMethod,
    chunkSize,
    setChunkSize,
    chunkOverlap,
    setChunkOverlap,
    availableMethods
  } = useChunkingConfig();

  // Vector store configuration
  const {
    backend,
    connectionStatus,
    embeddingModel,
    isLoading: isVectorStoreLoading,
    fetchConfig,
    checkHealth
  } = useVectorStoreConfig();

  // Document management
  const {
    documents,
    stats,
    isLoading: isDocumentsLoading,
    uploadProgress,
    isUploading,
    uploadDocument,
    deleteDocument,
    refreshAll
  } = useDocumentManager();

  // Handle document upload
  const handleUpload = useCallback(async (file, config) => {
    try {
      const result = await uploadDocument(file, {
        chunkingMethod: config.method,
        chunkSize: config.chunkSize,
        chunkOverlap: config.chunkOverlap
      });
      toast.success(`Successfully uploaded "${file.name}"`, {
        description: `${result.chunks_added || 0} chunks created`
      });
      return result;
    } catch (err) {
      toast.error("Upload failed", {
        description: err.message
      });
      throw err;
    }
  }, [uploadDocument]);

  // Handle document deletion
  const handleDelete = useCallback(async (filename) => {
    try {
      await deleteDocument(filename);
      toast.success("Document deleted", {
        description: `"${filename}" has been removed`
      });
    } catch (err) {
      toast.error("Delete failed", {
        description: err.message
      });
      throw err;
    }
  }, [deleteDocument]);

  // Handle vector store refresh
  const handleVectorStoreRefresh = useCallback(async () => {
    await fetchConfig();
    await checkHealth();
  }, [fetchConfig, checkHealth]);

  return (
    <div className="flex flex-col h-full bg-background">
      <Toaster position="top-right" richColors />

      {/* Header */}
      <header className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-xl font-semibold">Document Manager</h1>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="p-4 border-b">
        <DocumentStatsBar
          stats={stats}
          isLoading={isDocumentsLoading}
          onRefresh={refreshAll}
        />
      </div>

      {/* Main Content - Two Column Layout */}
      <div className="flex-1 overflow-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
          {/* Left Column - Configuration & Upload */}
          <div className="space-y-6">
            <ChunkingConfigCard
              method={chunkingMethod}
              onMethodChange={setChunkingMethod}
              chunkSize={chunkSize}
              onChunkSizeChange={setChunkSize}
              chunkOverlap={chunkOverlap}
              onChunkOverlapChange={setChunkOverlap}
              availableMethods={availableMethods}
            />

            <VectorStoreStatusCard
              backend={backend}
              connectionStatus={connectionStatus}
              embeddingModel={embeddingModel}
              onRefresh={handleVectorStoreRefresh}
              isLoading={isVectorStoreLoading}
            />

            <DocumentUploadCard
              onUpload={handleUpload}
              uploadProgress={uploadProgress}
              isUploading={isUploading}
              chunkingConfig={{
                method: chunkingMethod,
                chunkSize,
                chunkOverlap
              }}
            />
          </div>

          {/* Right Column - Document List */}
          <div>
            <DocumentSourceList
              documents={documents}
              onDeleteDocument={handleDelete}
              isLoading={isDocumentsLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentManager;
