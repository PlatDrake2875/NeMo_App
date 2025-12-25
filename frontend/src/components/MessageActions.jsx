import PropTypes from "prop-types";
import { Copy, Pencil, RotateCcw, Trash2, Check } from "lucide-react";
import { useState, useCallback } from "react";
import { Button } from "./ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

/**
 * MessageActions - Hover action bar for chat messages
 * @param {Object} props
 * @param {string} props.role - 'user' or 'assistant'
 * @param {string} props.content - Message content for copying
 * @param {function} props.onEdit - Handler for edit action (user only)
 * @param {function} props.onDelete - Handler for delete action
 * @param {function} props.onRegenerate - Handler for regenerate action (assistant only)
 * @param {boolean} props.isDeleting - Whether delete is in progress
 * @param {boolean} props.isRegenerating - Whether regeneration is in progress
 * @param {boolean} props.disabled - Disable all actions
 */
export function MessageActions({
  role,
  content,
  onEdit,
  onDelete,
  onRegenerate,
  isDeleting = false,
  isRegenerating = false,
  disabled = false
}) {
  const [copied, setCopied] = useState(false);
  const isUser = role === "user";

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }, [content]);

  return (
    <div className="flex items-center gap-1 bg-background border rounded-md shadow-sm p-0.5">
      {/* Copy Action */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleCopy}
            disabled={disabled}
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-green-500" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p>{copied ? "Copied!" : "Copy"}</p>
        </TooltipContent>
      </Tooltip>

      {/* Edit Action - User only */}
      {isUser && onEdit && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={onEdit}
              disabled={disabled}
            >
              <Pencil className="h-3.5 w-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>Edit</p>
          </TooltipContent>
        </Tooltip>
      )}

      {/* Regenerate Action - Assistant only */}
      {!isUser && onRegenerate && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={onRegenerate}
              disabled={disabled || isRegenerating}
            >
              <RotateCcw className={`h-3.5 w-3.5 ${isRegenerating ? "animate-spin" : ""}`} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>{isRegenerating ? "Regenerating..." : "Regenerate"}</p>
          </TooltipContent>
        </Tooltip>
      )}

      {/* Delete Action */}
      {onDelete && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-destructive hover:text-destructive"
              onClick={onDelete}
              disabled={disabled || isDeleting}
            >
              <Trash2 className={`h-3.5 w-3.5 ${isDeleting ? "animate-pulse" : ""}`} />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>{isDeleting ? "Deleting..." : "Delete"}</p>
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

MessageActions.propTypes = {
  role: PropTypes.oneOf(["user", "assistant", "bot", "system"]).isRequired,
  content: PropTypes.string.isRequired,
  onEdit: PropTypes.func,
  onDelete: PropTypes.func,
  onRegenerate: PropTypes.func,
  isDeleting: PropTypes.bool,
  isRegenerating: PropTypes.bool,
  disabled: PropTypes.bool
};
