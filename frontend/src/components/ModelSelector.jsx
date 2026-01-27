/**
 * ModelSelector - Dropdown for selecting and switching vLLM models.
 *
 * Features:
 * - Shows currently loaded model
 * - Lists cached models available for switching
 * - Triggers model switch with confirmation dialog
 * - Disabled during switch operations
 */

import { useState, useId } from "react";
import {
  ChevronDown,
  AlertCircle,
  Loader2,
  RefreshCw,
  HardDrive,
  Zap,
} from "lucide-react";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "./ui/dropdown-menu";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";
import { ModelSwitchDialog } from "./ModelSwitchDialog";

export function ModelSelector({
  availableModels = [],
  cachedModels = [],
  selectedModel,
  onModelSelect,
  modelsLoading,
  modelsError,
  onRefreshModels,
  switchStatus,
  onSwitchModel,
  onSwitchComplete,
  disabled,
}) {
  const selectId = useId();
  const [showSwitchDialog, setShowSwitchDialog] = useState(false);
  const [targetModel, setTargetModel] = useState(null);

  // Is a switch currently in progress?
  const isSwitching =
    switchStatus &&
    ["pending", "checking", "downloading", "stopping", "starting", "loading"].includes(
      switchStatus.status
    );

  // Format model name for display (remove org prefix)
  const formatModelName = (model) => {
    if (!model) return "No model";
    const name = model.split("/").pop();
    // Also remove :latest or similar suffixes for cleaner display
    return name.split(":")[0];
  };

  // Get short model name (first part only)
  const getShortName = (model) => {
    const name = formatModelName(model);
    // For long names, truncate
    if (name.length > 20) {
      return name.substring(0, 17) + "...";
    }
    return name;
  };

  // Handle clicking on a cached model to switch
  const handleCachedModelClick = (modelId) => {
    // Don't switch if already selected or switching
    if (modelId === selectedModel || isSwitching) return;

    setTargetModel(modelId);
    setShowSwitchDialog(true);
  };

  // Handle switch dialog completion
  const handleSwitchComplete = () => {
    onSwitchComplete?.();
    setTargetModel(null);
  };

  // Check if a cached model is currently loaded
  const isModelLoaded = (modelId) => {
    return availableModels.some(
      (m) => m === modelId || m.includes(modelId) || modelId.includes(m)
    );
  };

  return (
    <>
      <div className="flex items-center gap-1">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className={cn(
                "min-w-[180px] justify-between",
                isSwitching && "animate-pulse"
              )}
              disabled={modelsLoading || disabled || isSwitching}
            >
              <span className="flex items-center gap-2 truncate">
                {modelsLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading...
                  </>
                ) : modelsError ? (
                  <>
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    Error
                  </>
                ) : isSwitching ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Switching...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 text-green-500" />
                    {getShortName(selectedModel)}
                  </>
                )}
              </span>
              <ChevronDown className="h-4 w-4 ml-2 flex-shrink-0" />
            </Button>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="start" className="w-[280px]">
            {/* Currently loaded models */}
            <DropdownMenuLabel className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-green-500" />
              Active Model
            </DropdownMenuLabel>
            {availableModels.length > 0 ? (
              availableModels.map((model) => (
                <DropdownMenuItem
                  key={model}
                  onClick={() => onModelSelect?.(model)}
                  className={cn(
                    "flex items-center justify-between",
                    model === selectedModel && "bg-accent"
                  )}
                >
                  <span className="truncate">{formatModelName(model)}</span>
                  {model === selectedModel && (
                    <Badge variant="secondary" className="ml-2 text-xs">
                      Selected
                    </Badge>
                  )}
                </DropdownMenuItem>
              ))
            ) : (
              <DropdownMenuItem disabled>
                <span className="text-muted-foreground">No models loaded</span>
              </DropdownMenuItem>
            )}

            {/* Cached models (available to switch to) */}
            {cachedModels.length > 0 && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuLabel className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                  Cached Models (click to switch)
                </DropdownMenuLabel>
                {cachedModels
                  .filter((cm) => !isModelLoaded(cm.model_id))
                  .map((cached) => (
                    <DropdownMenuItem
                      key={cached.model_id}
                      onClick={() => handleCachedModelClick(cached.model_id)}
                      disabled={isSwitching}
                      className="flex items-center justify-between"
                    >
                      <span className="truncate">
                        {formatModelName(cached.model_id)}
                      </span>
                      <span className="text-xs text-muted-foreground ml-2">
                        {cached.size_gb?.toFixed(1)}GB
                      </span>
                    </DropdownMenuItem>
                  ))}
                {cachedModels.filter((cm) => !isModelLoaded(cm.model_id))
                  .length === 0 && (
                  <DropdownMenuItem disabled>
                    <span className="text-muted-foreground text-sm">
                      All cached models are loaded
                    </span>
                  </DropdownMenuItem>
                )}
              </>
            )}

            {/* Refresh button */}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => onRefreshModels?.()}
              disabled={modelsLoading}
            >
              <RefreshCw
                className={cn("h-4 w-4 mr-2", modelsLoading && "animate-spin")}
              />
              Refresh Models
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Show error tooltip if there's an error */}
        {modelsError && (
          <div className="text-xs text-destructive flex items-center gap-1">
            <AlertCircle className="h-3 w-3" />
            <span className="hidden sm:inline" title={modelsError}>
              Error
            </span>
          </div>
        )}
      </div>

      {/* Model Switch Dialog */}
      <ModelSwitchDialog
        open={showSwitchDialog}
        onOpenChange={setShowSwitchDialog}
        targetModel={targetModel}
        currentModel={selectedModel}
        onComplete={handleSwitchComplete}
      />
    </>
  );
}

export default ModelSelector;
