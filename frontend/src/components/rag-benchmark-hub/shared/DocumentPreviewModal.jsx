import { useEffect, useState, useRef } from "react";
import PropTypes from "prop-types";
import { Document, Page, pdfjs } from "react-pdf";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import { Button } from "../../ui/button";
import { ScrollArea } from "../../ui/scroll-area";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Loader2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getApiBaseUrl } from "../../../lib/api-config";
import { useTheme } from "../../../hooks/useTheme";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export function DocumentPreviewModal({ file, onClose }) {
  const { isDarkMode } = useTheme();
  const [content, setContent] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // PDF state
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1);

  // Ref to track blob URL for proper cleanup (avoids stale closure)
  const blobUrlRef = useRef(null);
  // Ref to track if component is mounted (for StrictMode double-invoke handling)
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;

    const fetchContent = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `${getApiBaseUrl()}/api/raw-datasets/${file.datasetId}/files/${file.id}/preview`
        );

        if (!response.ok) {
          throw new Error("Failed to fetch file content");
        }

        if (file.file_type === "pdf") {
          const blob = await response.blob();
          // Only update state if still mounted
          if (!isMountedRef.current) {
            return;
          }
          const url = URL.createObjectURL(blob);
          blobUrlRef.current = url;
          setContent(url);
        } else {
          const text = await response.text();
          // Only update state if still mounted
          if (!isMountedRef.current) {
            return;
          }
          setContent(text);
        }
      } catch (err) {
        if (!isMountedRef.current) {
          return;
        }
        console.error("Error fetching file:", err);
        setError(err.message);
      } finally {
        if (isMountedRef.current) {
          setIsLoading(false);
        }
      }
    };

    fetchContent();

    return () => {
      isMountedRef.current = false;
      // Clear content first to prevent react-pdf from rendering with stale URL
      setContent(null);
      // Then revoke the blob URL using ref (avoids stale closure)
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [file]);

  const handleDownload = () => {
    const downloadUrl = `${getApiBaseUrl()}/api/raw-datasets/${file.datasetId}/files/${file.id}/download`;
    window.open(downloadUrl, "_blank");
  };

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPageNumber(1);
  };

  const getLanguage = () => {
    switch (file.file_type) {
      case "json":
        return "json";
      case "md":
        return "markdown";
      case "csv":
        return "csv";
      default:
        return "text";
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-96">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-96 text-destructive">
          {error}
        </div>
      );
    }

    if (file.file_type === "pdf") {
      // Guard against rendering with null/destroyed content
      if (!content) {
        return null;
      }
      return (
        <div className="flex flex-col items-center">
          {/* PDF Controls */}
          <div className="flex items-center gap-4 mb-4 pb-4 border-b w-full justify-center">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm w-16 text-center">
                {Math.round(scale * 100)}%
              </span>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setScale((s) => Math.min(2, s + 0.25))}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
                disabled={pageNumber <= 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm w-24 text-center">
                Page {pageNumber} of {numPages || "?"}
              </span>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setPageNumber((p) => Math.min(numPages || 1, p + 1))}
                disabled={pageNumber >= (numPages || 1)}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* PDF Document */}
          <ScrollArea className="h-[60vh] w-full">
            <div className="flex justify-center">
              <Document
                key={content}
                file={content}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={
                  <div className="flex items-center justify-center h-96">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                }
                error={
                  <div className="text-destructive text-center p-4">
                    Failed to load PDF
                  </div>
                }
              >
                <Page
                  pageNumber={pageNumber}
                  scale={scale}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                />
              </Document>
            </div>
          </ScrollArea>
        </div>
      );
    }

    // Text-based files with syntax highlighting
    return (
      <ScrollArea className="h-[60vh]">
        <SyntaxHighlighter
          language={getLanguage()}
          style={isDarkMode ? oneDark : oneLight}
          customStyle={{
            margin: 0,
            borderRadius: "0.375rem",
            fontSize: "0.875rem",
          }}
          wrapLines
          wrapLongLines
        >
          {content || ""}
        </SyntaxHighlighter>
      </ScrollArea>
    );
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="pr-8 truncate">{file.filename}</DialogTitle>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>
          <DialogDescription className="sr-only">
            Preview of {file.filename}
          </DialogDescription>
        </DialogHeader>
        {renderContent()}
      </DialogContent>
    </Dialog>
  );
}

DocumentPreviewModal.propTypes = {
  file: PropTypes.shape({
    id: PropTypes.number.isRequired,
    datasetId: PropTypes.number.isRequired,
    filename: PropTypes.string.isRequired,
    file_type: PropTypes.string.isRequired,
  }).isRequired,
  onClose: PropTypes.func.isRequired,
};

export default DocumentPreviewModal;
