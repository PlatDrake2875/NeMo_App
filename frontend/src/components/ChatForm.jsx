import { useEffect, useId, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Send, Square, Paperclip, X } from "lucide-react";
import { cn } from "../lib/utils";
import { useFileAttachments } from "../hooks/useFileAttachments";
import { AttachmentPreview } from "./AttachmentPreview";
import { FileDropZone } from "./FileDropZone";
import { Alert } from "./ui/alert";

export function ChatForm({ onSubmit, disabled, isSubmitting, onStopGeneration }) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const chatInputId = useId();

  const {
    attachments,
    isUploading,
    uploadError,
    addFiles,
    removeAttachment,
    clearAttachments,
    getAttachmentsForMessage,
    hasAttachments,
    allowedTypes,
  } = useFileAttachments();

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const scrollHeight = textarea.scrollHeight;
      textarea.style.height = `${Math.min(scrollHeight, 200)}px`; // Max height 200px
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((!query.trim() && !hasAttachments) || disabled) return;

    // Get attachments metadata for the message
    const messageAttachments = hasAttachments ? getAttachmentsForMessage() : null;

    // Submit with query and attachments
    onSubmit(query, messageAttachments);

    // Clear form
    setQuery("");
    clearAttachments();
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      addFiles(files);
    }
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFilesDropped = (files) => {
    addFiles(files);
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <FileDropZone
      onFilesDropped={handleFilesDropped}
      disabled={disabled}
      acceptedTypes={allowedTypes}
      className="border-t bg-background"
    >
      <div className="px-4 py-4">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto"
          autoComplete="off"
        >
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={allowedTypes.join(",")}
            onChange={handleFileSelect}
            className="hidden"
            aria-hidden="true"
          />

          {/* Attachment Preview */}
          {hasAttachments && (
            <AttachmentPreview
              attachments={attachments}
              onRemove={removeAttachment}
              onClear={clearAttachments}
              disabled={isSubmitting}
              className="mb-3"
            />
          )}

          {/* Upload Error */}
          {uploadError && (
            <Alert variant="destructive" className="mb-3 py-2 text-sm">
              <div className="flex items-center justify-between">
                <span>{uploadError}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5"
                  onClick={() => {}}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            </Alert>
          )}

          <div className="flex gap-2 items-end">
            {/* Attach Button */}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={openFileDialog}
              disabled={disabled || isUploading}
              className="h-[52px] w-[52px] shrink-0"
              aria-label="Attach files"
              title="Attach files (images, PDF, text)"
            >
              <Paperclip className="h-5 w-5" />
            </Button>

            <div className="flex-1 relative">
              <label htmlFor={chatInputId} className="sr-only">
                Type your message (Shift + Enter for new line)
              </label>
              <textarea
                id={chatInputId}
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  disabled
                    ? "Select a model to chat..."
                    : hasAttachments
                      ? "Add a message or just send the files..."
                      : "Send a message... (Shift + Enter for new line)"
                }
                aria-label="Chat input"
                rows="1"
                disabled={disabled}
                className={cn(
                  "w-full resize-none rounded-lg border border-input bg-background px-4 py-3",
                  "text-sm placeholder:text-muted-foreground",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  "disabled:cursor-not-allowed disabled:opacity-50",
                  "min-h-[52px] max-h-[200px]",
                  "transition-colors"
                )}
              />
            </div>

            {isSubmitting ? (
              <Button
                type="button"
                size="icon"
                onClick={onStopGeneration}
                className="h-[52px] w-[52px] shrink-0 bg-destructive hover:bg-destructive/90"
                aria-label="Stop generation"
              >
                <Square className="h-5 w-5" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon"
                disabled={(!query.trim() && !hasAttachments) || disabled}
                className="h-[52px] w-[52px] shrink-0"
                aria-label="Send message"
              >
                <Send className="h-5 w-5" />
              </Button>
            )}
          </div>

          <div className="mt-2 text-xs text-muted-foreground text-center">
            Press <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">Enter</kbd> to send,{" "}
            <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">Shift + Enter</kbd> for new line
            {" "}• Drag & drop files or click <Paperclip className="inline h-3 w-3" /> to attach
          </div>
        </form>
      </div>
    </FileDropZone>
  );
}
