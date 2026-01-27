/**
 * ModelSwitchBanner - Global banner shown during model switching.
 *
 * Displays at the top of the page when a model switch is in progress,
 * showing progress and current step across all views.
 */

import { Loader2, XCircle } from "lucide-react";
import { Progress } from "./ui/progress";
import { Button } from "./ui/button";

export function ModelSwitchBanner({ switchStatus, onCancel }) {
  // Only show for in-progress statuses
  const inProgressStatuses = [
    "pending",
    "checking",
    "downloading",
    "stopping",
    "starting",
    "loading",
  ];

  if (!switchStatus || !inProgressStatuses.includes(switchStatus.status)) {
    return null;
  }

  const formatModelName = (model) => {
    if (!model) return "model";
    return model.split("/").pop();
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-amber-100 dark:bg-amber-900/80 border-b border-amber-300 dark:border-amber-700 px-4 py-2 shadow-sm">
      <div className="max-w-6xl mx-auto flex items-center gap-4">
        <Loader2 className="h-4 w-4 animate-spin text-amber-700 dark:text-amber-300 flex-shrink-0" />

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="font-medium text-amber-800 dark:text-amber-200 truncate">
              Switching to {formatModelName(switchStatus.to_model)}...
            </span>
            <span className="text-amber-600 dark:text-amber-400 ml-2 flex-shrink-0">
              {switchStatus.progress}%
            </span>
          </div>
          <Progress
            value={switchStatus.progress}
            className="h-1.5 bg-amber-200 dark:bg-amber-800"
          />
          <p className="text-xs text-amber-600 dark:text-amber-400 mt-1 truncate">
            {switchStatus.current_step || "Processing..."}
          </p>
        </div>

        {onCancel && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCancel}
            className="text-amber-700 dark:text-amber-300 hover:text-amber-900 dark:hover:text-amber-100 hover:bg-amber-200 dark:hover:bg-amber-800 flex-shrink-0"
          >
            <XCircle className="h-4 w-4 mr-1" />
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
}

export default ModelSwitchBanner;
