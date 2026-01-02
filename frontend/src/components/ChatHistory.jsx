import { useLayoutEffect, useRef } from "react";
import { ScrollArea } from "./ui/scroll-area";
import { MessageBubble } from "./MessageBubble";
import { LoadingIndicator } from "./LoadingIndicator";
import { cn } from "../lib/utils";

/**
 * Determines if messages should be grouped (from same sender within time threshold)
 * @param {Object} currentMsg - Current message
 * @param {Object} prevMsg - Previous message
 * @param {number} timeThreshold - Max milliseconds between messages to group (default: 2 minutes)
 * @returns {boolean} Whether messages should be grouped
 */
function shouldGroupMessages(currentMsg, prevMsg, timeThreshold = 120000) {
  if (!prevMsg || !currentMsg) return false;
  if (prevMsg.sender !== currentMsg.sender) return false;

  // Check timestamp proximity if available
  if (currentMsg.timestamp && prevMsg.timestamp) {
    const timeDiff = new Date(currentMsg.timestamp) - new Date(prevMsg.timestamp);
    return timeDiff < timeThreshold;
  }

  return true; // Group by default if no timestamps
}

export const ChatHistory = ({
  chatHistory,
  isLoading = false,
  isSubmitting = false,
  onEditMessage,
  onDeleteMessage,
  onRegenerateMessage
}) => {
  const endOfMessagesRef = useRef(null);
  const scrollAreaRef = useRef(null);

  useLayoutEffect(() => {
    if (!endOfMessagesRef.current) return;

    // Always scroll to bottom on new messages
    endOfMessagesRef.current.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [chatHistory, isLoading]);

  return (
    <ScrollArea className="flex-1 h-full scrollbar-thin" ref={scrollAreaRef}>
      <div className="flex flex-col">
        {!Array.isArray(chatHistory) || chatHistory.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[400px]">
            <div className="text-center space-y-3 max-w-md px-6">
              <div className="text-5xl mb-4">💬</div>
              <h3 className="text-lg font-semibold text-foreground">
                Start a conversation
              </h3>
              <p className="text-sm text-muted-foreground">
                Send a message to begin chatting with the AI assistant
              </p>
            </div>
          </div>
        ) : (
          <>
            {chatHistory.map((entry, index) => {
              // Validate entry
              if (
                typeof entry !== "object" ||
                entry === null ||
                !entry.sender ||
                !entry.text
              ) {
                console.warn("Skipping invalid chat history entry:", entry);
                return null;
              }

              // Determine if this message should be grouped with the previous one
              const prevEntry = index > 0 ? chatHistory[index - 1] : null;
              const isGrouped = shouldGroupMessages(entry, prevEntry);

              // Map sender to role
              const role = entry.sender === "user" ? "user" : "assistant";

              // Check for error messages
              const isError = entry.text.startsWith("⚠️ Error:");

              return (
                <MessageBubble
                  key={entry.id || `msg-${index}`}
                  role={role}
                  content={entry.text}
                  timestamp={entry.timestamp}
                  isGrouped={isGrouped}
                  messageId={entry.id}
                  onEdit={onEditMessage}
                  onDelete={onDeleteMessage}
                  onRegenerate={onRegenerateMessage}
                  showActions={!entry.isLoading}
                  isSubmitting={isSubmitting}
                  className={cn(
                    isError && "bg-destructive/10 border-l-4 border-l-destructive"
                  )}
                />
              );
            })}
          </>
        )}

        {/* Show loading indicator when assistant is responding */}
        {isLoading && <LoadingIndicator />}

        {/* Scroll anchor */}
        <div ref={endOfMessagesRef} className="h-4" />
      </div>
    </ScrollArea>
  );
};
