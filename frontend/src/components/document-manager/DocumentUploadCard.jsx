import PropTypes from "prop-types";
import { useCallback, useState } from "react";
import { FileText, Upload, X } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import { Badge } from "../ui/badge";

export function DocumentUploadCard({
  onUpload,
  uploadProgress,
  isUploading,
  chunkingConfig
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf") ||
          file.type === "text/plain" || file.name.endsWith(".txt")) {
        setSelectedFile(file);
      }
    }
  }, []);

  const handleFileSelect = useCallback((e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;
    try {
      await onUpload(selectedFile, chunkingConfig);
      setSelectedFile(null);
    } catch (err) {
      console.error("Upload failed:", err);
    }
  }, [selectedFile, onUpload, chunkingConfig]);

  const handleClearFile = useCallback(() => {
    setSelectedFile(null);
  }, []);

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Upload Document</CardTitle>
        <CardDescription>
          Upload PDF or text files to index in the vector store
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer
            ${isDragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-primary/50"
            }
            ${isUploading ? "pointer-events-none opacity-50" : ""}
          `}
          onClick={() => !isUploading && document.getElementById("file-upload")?.click()}
        >
          <input
            id="file-upload"
            type="file"
            accept=".pdf,.txt"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />
          <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
          <p className="text-sm text-muted-foreground">
            Drag and drop a file here, or click to select
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Supported: PDF, TXT
          </p>
        </div>

        {/* Selected File Preview */}
        {selectedFile && (
          <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
            <div className="flex items-center gap-3">
              <FileText className="h-8 w-8 text-primary" />
              <div>
                <p className="text-sm font-medium truncate max-w-[200px]">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClearFile}
              disabled={isUploading}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Chunking Config Preview */}
        {selectedFile && chunkingConfig && (
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              Method: {chunkingConfig.method}
            </Badge>
            <Badge variant="secondary">
              Size: {chunkingConfig.chunkSize}
            </Badge>
            <Badge variant="secondary">
              Overlap: {chunkingConfig.chunkOverlap}
            </Badge>
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} />
          </div>
        )}

        {/* Upload Button */}
        <Button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="w-full"
        >
          {isUploading ? "Uploading..." : "Upload & Index"}
        </Button>
      </CardContent>
    </Card>
  );
}

DocumentUploadCard.propTypes = {
  onUpload: PropTypes.func.isRequired,
  uploadProgress: PropTypes.number,
  isUploading: PropTypes.bool,
  chunkingConfig: PropTypes.shape({
    method: PropTypes.string,
    chunkSize: PropTypes.number,
    chunkOverlap: PropTypes.number
  })
};
