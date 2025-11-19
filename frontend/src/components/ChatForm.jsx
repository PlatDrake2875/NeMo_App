import { useEffect, useId, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Send, Square } from "lucide-react";
import { cn } from "../lib/utils";

export function ChatForm({ onSubmit, disabled, isSubmitting, onStopGeneration }) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef(null);
  const chatInputId = useId();

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
    if (!query.trim() || disabled) return;
    onSubmit(query);
    setQuery("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  return (
    <div className="border-t bg-background px-4 py-4">
      <form
        onSubmit={handleSubmit}
        className="max-w-4xl mx-auto"
        autoComplete="off"
      >
        <div className="flex gap-2 items-end">
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
              disabled={!query.trim() || disabled}
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
        </div>
      </form>
    </div>
  );
}
