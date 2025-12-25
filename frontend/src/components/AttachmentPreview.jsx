// frontend/src/components/AttachmentPreview.jsx
import PropTypes from "prop-types";
import { X, FileText, Image, File, Loader2 } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

/**
 * Get the appropriate icon for a file type
 */
function getFileIcon(mimeType) {
  if (mimeType.startsWith("image/")) {
    return Image;
  }
  if (mimeType === "application/pdf" || mimeType.startsWith("text/")) {
    return FileText;
  }
  return File;
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Single attachment preview item
 */
function AttachmentItem({ attachment, onRemove, disabled }) {
  const { id, file, preview, status } = attachment;
  const FileIcon = getFileIcon(file.type);
  const isImage = file.type.startsWith("image/");
  const isUploading = status === "uploading";
  const hasError = status === "error";

  return (
    <div
      className={cn(
        "relative group flex items-center gap-2 p-2 rounded-lg border bg-background",
        "transition-all hover:border-primary/50",
        hasError && "border-destructive bg-destructive/10",
        isUploading && "opacity-70"
      )}
    >
      {/* Preview/Icon */}
      <div className="relative flex-shrink-0 w-10 h-10 rounded overflow-hidden bg-muted flex items-center justify-center">
        {isImage && preview ? (
          <img
            src={preview}
            alt={file.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <FileIcon className="h-5 w-5 text-muted-foreground" />
        )}
        {isUploading && (
          <div className="absolute inset-0 bg-background/60 flex items-center justify-center">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
          </div>
        )}
      </div>

      {/* File Info */}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{file.name}</p>
        <p className="text-xs text-muted-foreground">
          {formatFileSize(file.size)}
        </p>
      </div>

      {/* Remove Button */}
      {!disabled && (
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity",
            "hover:bg-destructive/10 hover:text-destructive"
          )}
          onClick={() => onRemove(id)}
          aria-label={`Remove ${file.name}`}
        >
          <X className="h-3 w-3" />
        </Button>
      )}
    </div>
  );
}

AttachmentItem.propTypes = {
  attachment: PropTypes.shape({
    id: PropTypes.string.isRequired,
    file: PropTypes.object.isRequired,
    preview: PropTypes.string,
    status: PropTypes.oneOf(["pending", "uploading", "uploaded", "error"]),
  }).isRequired,
  onRemove: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
};

/**
 * AttachmentPreview - Display previews of attached files
 */
export function AttachmentPreview({
  attachments,
  onRemove,
  onClear,
  disabled = false,
  className,
}) {
  if (!attachments || attachments.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {attachments.length} file{attachments.length !== 1 ? "s" : ""} attached
        </span>
        {!disabled && onClear && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="h-6 text-xs text-muted-foreground hover:text-foreground"
          >
            Clear all
          </Button>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {attachments.map((attachment) => (
          <AttachmentItem
            key={attachment.id}
            attachment={attachment}
            onRemove={onRemove}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  );
}

AttachmentPreview.propTypes = {
  attachments: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      file: PropTypes.object.isRequired,
      preview: PropTypes.string,
      status: PropTypes.oneOf(["pending", "uploading", "uploaded", "error"]),
    })
  ).isRequired,
  onRemove: PropTypes.func.isRequired,
  onClear: PropTypes.func,
  disabled: PropTypes.bool,
  className: PropTypes.string,
};
