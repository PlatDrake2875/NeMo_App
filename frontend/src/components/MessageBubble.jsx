import * as React from "react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { cn } from "../lib/utils";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
 */
export function MessageBubble({
  role,
  content,
  timestamp,
  isGrouped = false,
  className
}) {
  const isUser = role === "user";
  const messageTime = timestamp ? new Date(timestamp) : new Date();

  return (
    <div
      className={cn(
        "group flex gap-3 px-4 py-2 hover:bg-accent/50 transition-colors",
        isGrouped && "mt-0.5",
        !isGrouped && "mt-4",
        className
      )}
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
    </div>
  );
}
