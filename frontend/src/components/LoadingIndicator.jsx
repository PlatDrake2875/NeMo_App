import * as React from "react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Bot } from "lucide-react";
import { cn } from "../lib/utils";

/**
 * TypingIndicator - Animated three-dot typing indicator
 */
export function TypingIndicator({ className }) {
  return (
    <div className={cn("flex gap-1", className)}>
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-typing" style={{ animationDelay: "0ms" }} />
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-typing" style={{ animationDelay: "200ms" }} />
      <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-typing" style={{ animationDelay: "400ms" }} />
    </div>
  );
}

/**
 * LoadingIndicator - Shows when assistant is thinking/responding
 */
export function LoadingIndicator({ className }) {
  return (
    <div className={cn("flex gap-3 px-4 py-2", className)}>
      {/* Avatar */}
      <div className="flex-shrink-0">
        <Avatar className="h-9 w-9 bg-secondary">
          <AvatarFallback className="bg-secondary text-secondary-foreground">
            <Bot className="h-5 w-5" />
          </AvatarFallback>
        </Avatar>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 mb-1">
          <span className="font-semibold text-sm text-foreground">
            Assistant
          </span>
        </div>
        <div className="flex items-center h-6">
          <TypingIndicator />
        </div>
      </div>
    </div>
  );
}

/**
 * MessageSkeleton - Skeleton loader for initial chat load
 */
export function MessageSkeleton({ count = 3 }) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex gap-3 px-4 py-2 animate-pulse">
          {/* Avatar skeleton */}
          <div className="flex-shrink-0">
            <div className="h-9 w-9 rounded-full bg-muted" />
          </div>

          {/* Content skeleton */}
          <div className="flex-1 min-w-0 space-y-2">
            <div className="flex items-center gap-2">
              <div className="h-4 w-16 bg-muted rounded" />
              <div className="h-3 w-12 bg-muted rounded" />
            </div>
            <div className="space-y-1.5">
              <div className="h-4 bg-muted rounded w-full" />
              <div className="h-4 bg-muted rounded w-5/6" />
            </div>
          </div>
        </div>
      ))}
    </>
  );
}
