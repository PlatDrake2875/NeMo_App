import PropTypes from "prop-types";
import { useState, useCallback, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "./ui/dialog";
import { Button } from "./ui/button";

/**
 * EditMessageModal - Modal for editing user messages
 * @param {Object} props
 * @param {boolean} props.open - Whether the modal is open
 * @param {function} props.onOpenChange - Handler for open state changes
 * @param {string} props.originalContent - The original message content
 * @param {function} props.onSave - Handler for saving the edited message
 */
export function EditMessageModal({
  open,
  onOpenChange,
  originalContent,
  onSave
}) {
  const [content, setContent] = useState(originalContent);
  const [isSaving, setIsSaving] = useState(false);

  // Reset content when modal opens with new content
  useEffect(() => {
    if (open) {
      setContent(originalContent);
    }
  }, [open, originalContent]);

  const handleSave = useCallback(async () => {
    if (!content.trim() || content === originalContent) {
      onOpenChange(false);
      return;
    }

    setIsSaving(true);
    try {
      await onSave(content.trim());
      onOpenChange(false);
    } catch (err) {
      console.error("Failed to save:", err);
    } finally {
      setIsSaving(false);
    }
  }, [content, originalContent, onSave, onOpenChange]);

  const handleKeyDown = useCallback((e) => {
    // Ctrl/Cmd + Enter to save
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSave();
    }
  }, [handleSave]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Edit Message</DialogTitle>
          <DialogDescription>
            Edit your message and submit again. The assistant will regenerate a response.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full min-h-[150px] p-3 border rounded-md resize-y bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Enter your message..."
            autoFocus
          />
          <p className="text-xs text-muted-foreground mt-2">
            Press Ctrl+Enter to save
          </p>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={isSaving || !content.trim() || content === originalContent}
          >
            {isSaving ? "Saving..." : "Save & Regenerate"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

EditMessageModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onOpenChange: PropTypes.func.isRequired,
  originalContent: PropTypes.string.isRequired,
  onSave: PropTypes.func.isRequired
};
