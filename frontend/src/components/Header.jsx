import { useId, useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
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
  Trash2,
  Download,
  MoreVertical,
  AlertCircle,
  Sparkles,
  ChevronDown,
  Plus,
  Database,
  FileJson,
  FileText,
  Upload,
  Settings,
} from "lucide-react";
import { cn } from "../lib/utils";
import { ModelDownloadDialog } from "./ModelDownloadDialog";
import { ImportConversationDialog } from "./ImportConversationDialog";
import { AdvancedSettingsDropdown } from "./AdvancedSettingsDropdown";
import { exportAsJSON, exportAsPDF } from "../utils/exportUtils";

export function Header({
  activeSessionName,
  activeSessionId,
  activeSession,
  chatHistory,
  clearChatHistory,
  onImportConversation,
  disabled,
  availableModels,
  selectedModel,
  onModelChange,
  modelsLoading,
  modelsError,
  onRefreshModels,
  // Advanced Settings props
  selectedDataset,
  onDatasetChange,
  isRagEnabled = true,
  onRagEnabledChange,
  isColbertEnabled = true,
  onColbertEnabledChange,
}) {
  const modelSelectId = useId();
  const title = activeSessionName || "Chat";
  const isHistoryEmpty =
    !Array.isArray(chatHistory) || chatHistory.length === 0;
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [vectorStoreBackend, setVectorStoreBackend] = useState(null);

  // Fetch vector store configuration on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch("/api/config/vector-store");
        if (response.ok) {
          const data = await response.json();
          setVectorStoreBackend(data.backend);
        }
      } catch (error) {
        console.error("Failed to fetch vector store config:", error);
      }
    };
    fetchConfig();
  }, []);

  const handleDownloadComplete = (modelId) => {
    // Refresh the models list
    if (onRefreshModels) {
      onRefreshModels();
    }
  };

  const handleExportJSON = () => {
    if (activeSession && activeSessionId) {
      exportAsJSON(activeSession, activeSessionId, selectedModel);
    }
  };

  const handleExportPDF = () => {
    if (activeSession && activeSessionId) {
      exportAsPDF(activeSession, activeSessionId, selectedModel);
    }
  };

  const handleImport = async (importedData) => {
    if (onImportConversation) {
      await onImportConversation(importedData);
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
          {/* Vector Store Backend Badge */}
          {vectorStoreBackend && (
            <Badge
              variant={vectorStoreBackend === "qdrant" ? "default" : "secondary"}
              className="flex items-center gap-1 text-xs"
              title={`Vector Store: ${vectorStoreBackend}`}
            >
              <Database className="h-3 w-3" />
              {vectorStoreBackend === "qdrant" ? "Qdrant" : "PGVector"}
            </Badge>
          )}
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
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowAdvancedSettings(true)}
              disabled={disabled}
              aria-label="Advanced Settings"
              title="Advanced Settings"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>

          <Separator orientation="vertical" className="h-8" />

          {/* Quick Actions */}
          <div className="hidden md:flex items-center gap-1">
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

            {/* Export/Import Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Export or import conversation"
                  title="Export/Import"
                  disabled={disabled}
                >
                  <Download className="h-5 w-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel>Export</DropdownMenuLabel>
                <DropdownMenuItem
                  onClick={handleExportJSON}
                  disabled={isHistoryEmpty}
                >
                  <FileJson className="mr-2 h-4 w-4" />
                  Export as JSON
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={handleExportPDF}
                  disabled={isHistoryEmpty}
                >
                  <FileText className="mr-2 h-4 w-4" />
                  Export as PDF
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Import</DropdownMenuLabel>
                <DropdownMenuItem onClick={() => setShowImportDialog(true)}>
                  <Upload className="mr-2 h-4 w-4" />
                  Import Conversation
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
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
              <DropdownMenuItem
                onClick={clearChatHistory}
                disabled={disabled || isHistoryEmpty}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Clear Chat
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Export</DropdownMenuLabel>
              <DropdownMenuItem
                onClick={handleExportJSON}
                disabled={isHistoryEmpty}
              >
                <FileJson className="mr-2 h-4 w-4" />
                Export as JSON
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={handleExportPDF}
                disabled={isHistoryEmpty}
              >
                <FileText className="mr-2 h-4 w-4" />
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setShowImportDialog(true)}>
                <Upload className="mr-2 h-4 w-4" />
                Import Conversation
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

      {/* Import Conversation Dialog */}
      <ImportConversationDialog
        open={showImportDialog}
        onOpenChange={setShowImportDialog}
        onImport={handleImport}
      />

      {/* Advanced Settings Dropdown */}
      <AdvancedSettingsDropdown
        open={showAdvancedSettings}
        onOpenChange={setShowAdvancedSettings}
        selectedDataset={selectedDataset}
        onDatasetChange={onDatasetChange}
        isRagEnabled={isRagEnabled}
        onRagEnabledChange={onRagEnabledChange}
        isColbertEnabled={isColbertEnabled}
        onColbertEnabledChange={onColbertEnabledChange}
      />
    </header>
  );
}
