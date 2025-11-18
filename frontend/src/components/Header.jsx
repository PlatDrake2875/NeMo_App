import { useId, useState } from "react";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "./ui/dropdown-menu";
import { Separator } from "./ui/separator";
import {
  Moon,
  Sun,
  Trash2,
  Download,
  MoreVertical,
  AlertCircle,
  Sparkles,
  ChevronDown,
  Plus,
} from "lucide-react";
import { cn } from "../lib/utils";
import { ModelDownloadDialog } from "./ModelDownloadDialog";

export function Header({
  activeSessionName,
  chatHistory,
  clearChatHistory,
  downloadChatHistory,
  disabled,
  isDarkMode,
  toggleTheme,
  availableModels,
  selectedModel,
  onModelChange,
  modelsLoading,
  modelsError,
  onRefreshModels,
}) {
  const modelSelectId = useId();
  const title = activeSessionName || "Chat";
  const isHistoryEmpty =
    !Array.isArray(chatHistory) || chatHistory.length === 0;
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);

  const handleDownloadComplete = (modelId) => {
    // Refresh the models list
    if (onRefreshModels) {
      onRefreshModels();
    }
  };

  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center justify-between px-4 py-3 gap-4">
        {/* Title Section */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <Sparkles className="h-5 w-5 text-primary flex-shrink-0" />
            <h1 className="text-lg font-semibold truncate">{title}</h1>
          </div>
        </div>

        {/* Controls Section */}
        <div className="flex items-center gap-2">
          {/* Model Selector with Download Button */}
          <div className="flex items-center gap-1">
            <div className="relative min-w-[180px]">
              <label htmlFor={modelSelectId} className="sr-only">
                Select Model
              </label>
              <div className="relative">
                <select
                  id={modelSelectId}
                  value={selectedModel}
                  onChange={onModelChange}
                  disabled={
                    modelsLoading ||
                    availableModels.length === 0 ||
                    !!modelsError ||
                    disabled
                  }
                  className={cn(
                    "w-full appearance-none rounded-md border border-input bg-background px-3 py-2 pr-8 text-sm",
                    "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                    "disabled:cursor-not-allowed disabled:opacity-50",
                    "transition-colors"
                  )}
                  aria-label="Select AI model"
                >
                  {modelsLoading && <option value="">Loading models...</option>}
                  {modelsError && <option value="">Error loading models</option>}
                  {!modelsLoading && !modelsError && availableModels.length === 0 && (
                    <option value="">No models found</option>
                  )}
                  {!modelsLoading &&
                    !modelsError &&
                    availableModels.map((modelName) => (
                      <option key={modelName} value={modelName}>
                        {modelName.split(":")[0]}
                      </option>
                    ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
              </div>
              {modelsError && (
                <div className="absolute right-0 top-full mt-1 flex items-center gap-1 text-xs text-destructive">
                  <AlertCircle className="h-3 w-3" />
                  <span title={modelsError}>Model error</span>
                </div>
              )}
            </div>
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowDownloadDialog(true)}
              disabled={disabled}
              aria-label="Download new model from HuggingFace"
              title="Download Model"
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>

          <Separator orientation="vertical" className="h-8" />

          {/* Quick Actions */}
          <div className="hidden md:flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label={`Switch to ${isDarkMode ? "light" : "dark"} mode`}
              title={`Switch to ${isDarkMode ? "Light" : "Dark"} Mode`}
              disabled={disabled}
            >
              {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={clearChatHistory}
              disabled={disabled || isHistoryEmpty}
              aria-label="Clear current chat history"
              title="Clear Chat"
            >
              <Trash2 className="h-5 w-5" />
            </Button>

            <Button
              variant="ghost"
              size="icon"
              onClick={downloadChatHistory}
              disabled={disabled || isHistoryEmpty}
              aria-label="Download current chat history"
              title="Download Chat"
            >
              <Download className="h-5 w-5" />
            </Button>
          </div>

          {/* Mobile Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon" aria-label="More options">
                <MoreVertical className="h-5 w-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={toggleTheme} disabled={disabled}>
                {isDarkMode ? (
                  <Sun className="mr-2 h-4 w-4" />
                ) : (
                  <Moon className="mr-2 h-4 w-4" />
                )}
                {isDarkMode ? "Light Mode" : "Dark Mode"}
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={clearChatHistory}
                disabled={disabled || isHistoryEmpty}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Clear Chat
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={downloadChatHistory}
                disabled={disabled || isHistoryEmpty}
              >
                <Download className="mr-2 h-4 w-4" />
                Download Chat
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Model Download Dialog */}
      <ModelDownloadDialog
        open={showDownloadDialog}
        onOpenChange={setShowDownloadDialog}
        onDownloadComplete={handleDownloadComplete}
      />
    </header>
  );
}
