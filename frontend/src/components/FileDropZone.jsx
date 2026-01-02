// frontend/src/components/FileDropZone.jsx
import PropTypes from "prop-types";
import { useState, useCallback, useRef } from "react";
import { Upload } from "lucide-react";
import { cn } from "../lib/utils";

/**
 * FileDropZone - A drop zone wrapper for file attachments
 * Wraps around content and shows an overlay when files are dragged over
 */
export function FileDropZone({
  children,
  onFilesDropped,
  disabled = false,
  acceptedTypes,
  className,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounterRef = useRef(0);

  const handleDragEnter = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (disabled) return;

      dragCounterRef.current++;
      if (dragCounterRef.current === 1) {
        // Check if the dragged items contain files
        const hasFiles = Array.from(e.dataTransfer.items || []).some(
          (item) => item.kind === "file"
        );
        if (hasFiles) {
          setIsDragOver(true);
        }
      }
    },
    [disabled]
  );

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();

    dragCounterRef.current--;
    if (dragCounterRef.current === 0) {
      setIsDragOver(false);
    }
  }, []);

  const handleDragOver = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (disabled) return;

      // Set the drop effect
      e.dataTransfer.dropEffect = "copy";
    },
    [disabled]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();

      dragCounterRef.current = 0;
      setIsDragOver(false);

      if (disabled) return;

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        onFilesDropped(files);
      }
    },
    [disabled, onFilesDropped]
  );

  return (
    <div
      className={cn("relative", className)}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {children}

      {/* Drop Overlay */}
      {isDragOver && !disabled && (
        <div
          className={cn(
            "absolute inset-0 z-50 flex flex-col items-center justify-center",
            "bg-primary/10 border-2 border-dashed border-primary rounded-lg",
            "pointer-events-none"
          )}
        >
          <div className="bg-background rounded-lg p-6 shadow-lg text-center">
            <Upload className="h-10 w-10 text-primary mx-auto mb-2" />
            <p className="text-sm font-medium text-foreground">
              Drop files to attach
            </p>
            {acceptedTypes && (
              <p className="text-xs text-muted-foreground mt-1">
                Accepted: images, PDF, text files
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

FileDropZone.propTypes = {
  children: PropTypes.node.isRequired,
  onFilesDropped: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  acceptedTypes: PropTypes.arrayOf(PropTypes.string),
  className: PropTypes.string,
};
