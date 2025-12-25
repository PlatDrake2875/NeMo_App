import * as React from "react";
import { useState } from "react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { cn } from "../lib/utils";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MessageActions } from "./MessageActions";
import { EditMessageModal } from "./EditMessageModal";

/**
 * Format timestamp for display
 * @param {Date} date - The date to format
 * @returns {string} Formatted time string
 */
function formatTime(date) {
  if (!date) return "";
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  // For older messages, show the actual time
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });
}

/**
 * MessageBubble - Modern Slack/Discord-style message component
 * @param {Object} props
 * @param {string} props.role - 'user' or 'assistant'
 * @param {string} props.content - Message content (supports markdown)
 * @param {Date} props.timestamp - Message timestamp
 * @param {boolean} props.isGrouped - Whether message is grouped with previous
 * @param {string} props.className - Additional CSS classes
 * @param {string} props.messageId - Unique message ID
 * @param {function} props.onEdit - Handler for editing user messages
 * @param {function} props.onDelete - Handler for deleting messages
 * @param {function} props.onRegenerate - Handler for regenerating assistant messages
 * @param {boolean} props.showActions - Whether to show action buttons
 * @param {boolean} props.isSubmitting - Whether a submission is in progress
 */
export function MessageBubble({
  role,
  content,
  timestamp,
  isGrouped = false,
  className,
  messageId,
  onEdit,
  onDelete,
  onRegenerate,
  showActions = true,
  isSubmitting = false
}) {
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const isUser = role === "user";
  const messageTime = timestamp ? new Date(timestamp) : new Date();

  // Handle edit save - edits the message and triggers regeneration
  const handleEditSave = async (newContent) => {
    if (onEdit && messageId) {
      await onEdit(messageId, newContent);
    }
  };

  // Handle delete
  const handleDelete = () => {
    if (onDelete && messageId) {
      onDelete(messageId);
    }
  };

  // Handle regenerate
  const handleRegenerate = () => {
    if (onRegenerate && messageId) {
      onRegenerate(messageId);
    }
  };

  return (
    <>
    <div
      className={cn(
        "group relative flex gap-3 px-4 py-2 hover:bg-accent/50 transition-colors",
        isGrouped && "mt-0.5",
        !isGrouped && "mt-4",
        className
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Avatar - only show if not grouped */}
      <div className="flex-shrink-0">
        {!isGrouped ? (
          <Avatar className={cn(
            "h-9 w-9",
            isUser ? "bg-primary" : "bg-secondary"
          )}>
            <AvatarFallback className={cn(
              "text-sm font-medium",
              isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
            )}>
              {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
            </AvatarFallback>
          </Avatar>
        ) : (
          <div className="w-9" /> /* Spacer for grouped messages */
        )}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        {/* Header - only show if not grouped */}
        {!isGrouped && (
          <div className="flex items-baseline gap-2 mb-1">
            <span className="font-semibold text-sm text-foreground">
              {isUser ? "You" : "Assistant"}
            </span>
            <span className="text-xs text-muted-foreground">
              {formatTime(messageTime)}
            </span>
          </div>
        )}

        {/* Message Text */}
        <div className={cn(
          "text-sm text-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none",
          "prose-p:my-1 prose-pre:my-2 prose-pre:bg-muted prose-pre:border prose-pre:border-border",
          "prose-code:text-foreground prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded",
          "prose-a:text-primary hover:prose-a:underline",
          "prose-ul:my-1 prose-ol:my-1"
        )}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
          </ReactMarkdown>
        </div>
      </div>

      {/* Timestamp on hover - shows for grouped messages */}
      {isGrouped && (
        <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-xs text-muted-foreground">
            {messageTime.toLocaleTimeString('en-US', {
              hour: 'numeric',
              minute: '2-digit',
              hour12: true
            })}
          </span>
        </div>
      )}

      {/* Message Actions - appears on hover */}
      {showActions && isHovered && !isSubmitting && (
        <div className="absolute top-0 right-4 -translate-y-1/2 z-10">
          <MessageActions
            role={role}
            content={content}
            onEdit={isUser && onEdit ? () => setIsEditModalOpen(true) : undefined}
            onDelete={onDelete ? handleDelete : undefined}
            onRegenerate={!isUser && onRegenerate ? handleRegenerate : undefined}
            disabled={isSubmitting}
          />
        </div>
      )}
    </div>

    {/* Edit Message Modal */}
    {isUser && onEdit && (
      <EditMessageModal
        open={isEditModalOpen}
        onOpenChange={setIsEditModalOpen}
        originalContent={content}
        onSave={handleEditSave}
      />
    )}
    </>
  );
}
