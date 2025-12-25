// frontend/src/components/MessageAttachments.jsx
import PropTypes from "prop-types";
import { FileText, Image, File, Download, ExternalLink } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

/**
 * Get the appropriate icon for a file type
 */
function getFileIcon(mimeType) {
  if (!mimeType) return File;
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
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Single attachment display in a message
 */
function AttachmentDisplay({ attachment, isUserMessage }) {
  const { name, type, size, preview, uploadedInfo } = attachment;
  const FileIcon = getFileIcon(type);
  const isImage = type?.startsWith("image/");
  const downloadUrl = uploadedInfo?.url;

  return (
    <div
      className={cn(
        "flex items-center gap-2 p-2 rounded-md",
        isUserMessage
          ? "bg-primary-foreground/10"
          : "bg-muted"
      )}
    >
      {/* Image Preview or Icon */}
      <div className="flex-shrink-0 w-10 h-10 rounded overflow-hidden bg-muted/50 flex items-center justify-center">
        {isImage && preview ? (
          <img
            src={preview}
            alt={name}
            className="w-full h-full object-cover cursor-pointer"
            onClick={() => window.open(preview, "_blank")}
          />
        ) : (
          <FileIcon className="h-5 w-5 text-muted-foreground" />
        )}
      </div>

      {/* File Info */}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{name}</p>
        {size && (
          <p className="text-xs text-muted-foreground">
            {formatFileSize(size)}
          </p>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-1">
        {downloadUrl && (
          <>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              asChild
              title="Download"
            >
              <a href={downloadUrl} download={name}>
                <Download className="h-3 w-3" />
              </a>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              asChild
              title="Open in new tab"
            >
              <a href={downloadUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-3 w-3" />
              </a>
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

AttachmentDisplay.propTypes = {
  attachment: PropTypes.shape({
    id: PropTypes.string,
    name: PropTypes.string.isRequired,
    type: PropTypes.string,
    size: PropTypes.number,
    preview: PropTypes.string,
    uploadedInfo: PropTypes.shape({
      url: PropTypes.string,
    }),
  }).isRequired,
  isUserMessage: PropTypes.bool,
};

/**
 * MessageAttachments - Display attachments within a message bubble
 */
export function MessageAttachments({ attachments, isUserMessage = false, className }) {
  if (!attachments || attachments.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-1 mt-2", className)}>
      {attachments.map((attachment, index) => (
        <AttachmentDisplay
          key={attachment.id || index}
          attachment={attachment}
          isUserMessage={isUserMessage}
        />
      ))}
    </div>
  );
}

MessageAttachments.propTypes = {
  attachments: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string,
      name: PropTypes.string.isRequired,
      type: PropTypes.string,
      size: PropTypes.number,
      preview: PropTypes.string,
      uploadedInfo: PropTypes.object,
    })
  ),
  isUserMessage: PropTypes.bool,
  className: PropTypes.string,
};
