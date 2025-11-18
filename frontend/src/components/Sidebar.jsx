import PropTypes from "prop-types";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";
import { Input } from "./ui/input";
import { Checkbox } from "./ui/checkbox";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import {
  MessageSquarePlus,
  FileText,
  Pencil,
  Trash2,
  Check,
  X,
  Upload,
  Play,
  FileUp,
  Loader2,
  CheckSquare,
  Square,
} from "lucide-react";
import { cn } from "../lib/utils";

const formatSessionIdFallback = (sessionId) => {
  if (!sessionId) return "Chat";
  return sessionId.replace(/-/g, " ").replace(/^./, (str) => str.toUpperCase());
};

export function Sidebar({
  sessions,
  activeSessionId,
  selectedModel,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
  onAutomateConversation,
  isSubmitting,
  automationError,
  isInitialized,
  onUploadPdf,
  isUploadingPdf,
  pdfUploadStatus,
  onViewDocuments,
}) {
  const sessionIds = isInitialized ? Object.keys(sessions) : [];
  const [automationJson, setAutomationJson] = useState(
    '{\n  "inputs": [\n    "Hello!",\n    "How are you?"\n  ]\n}'
  );
  const automationFileInputRef = useRef(null);
  const pdfFileInputRef = useRef(null);
  const [selectedPdfFile, setSelectedPdfFile] = useState(null);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingValue, setEditingValue] = useState("");
  const editInputRef = useRef(null);

  // Multi-select state
  const [selectedSessions, setSelectedSessions] = useState(new Set());
  const [isSelectionMode, setIsSelectionMode] = useState(false);

  const handleEditClick = (e, sessionId) => {
    e.stopPropagation();
    const currentName =
      sessions[sessionId]?.name || formatSessionIdFallback(sessionId);
    setEditingSessionId(sessionId);
    setEditingValue(currentName);
  };

  const handleSaveEdit = () => {
    if (editingSessionId && editingValue.trim()) {
      onRenameSession(editingSessionId, editingValue.trim());
    }
    setEditingSessionId(null);
    setEditingValue("");
  };

  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingValue("");
  };

  const handleInputChange = (e) => {
    setEditingValue(e.target.value);
  };

  const handleInputKeyDown = (e) => {
    if (e.key === "Enter") handleSaveEdit();
    else if (e.key === "Escape") handleCancelEdit();
  };

  useEffect(() => {
    if (editingSessionId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingSessionId]);

  // Multi-select handlers
  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode);
    setSelectedSessions(new Set());
  };

  const toggleSessionSelection = (sessionId) => {
    setSelectedSessions((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(sessionId)) {
        newSet.delete(sessionId);
      } else {
        newSet.add(sessionId);
      }
      return newSet;
    });
  };

  const selectAll = () => {
    setSelectedSessions(new Set(sessionIds));
  };

  const deselectAll = () => {
    setSelectedSessions(new Set());
  };

  const handleBulkDelete = () => {
    if (selectedSessions.size === 0) return;

    const count = selectedSessions.size;
    if (confirm(`Are you sure you want to delete ${count} chat${count > 1 ? 's' : ''}?`)) {
      selectedSessions.forEach((sessionId) => {
        onDeleteSession(sessionId);
      });
      setSelectedSessions(new Set());
      setIsSelectionMode(false);
    }
  };

  const handleAutomationSubmit = () => {
    if (!selectedModel) {
      alert("Please select a model first.");
      return;
    }
    if (!activeSessionId) {
      alert("Please select an active chat session to automate.");
      return;
    }
    const taskToPerform = "summarize_conversation";
    if (onAutomateConversation) {
      onAutomateConversation(automationJson, selectedModel, taskToPerform);
    }
  };

  const handleUploadJsonClick = () => {
    automationFileInputRef.current?.click();
  };

  const handleJsonFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.type !== "application/json") {
      alert("Please select a valid JSON file (.json)");
      if (automationFileInputRef.current)
        automationFileInputRef.current.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result;
        if (typeof text === "string") {
          JSON.parse(text);
          setAutomationJson(text);
        }
      } catch (error) {
        alert(`Error reading file: ${error.message}`);
      } finally {
        if (automationFileInputRef.current)
          automationFileInputRef.current.value = "";
      }
    };
    reader.onerror = () => {
      alert("Error reading file.");
      if (automationFileInputRef.current)
        automationFileInputRef.current.value = "";
    };
    reader.readAsText(file);
  };

  const handlePdfFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setSelectedPdfFile(file);
      if (pdfUploadStatus && onUploadPdf) onUploadPdf(null, true);
    } else {
      setSelectedPdfFile(null);
      if (file) alert("Please select a PDF file.");
      if (pdfFileInputRef.current) pdfFileInputRef.current.value = "";
    }
  };

  const handlePdfUploadClick = () => {
    if (selectedPdfFile && onUploadPdf) {
      onUploadPdf(selectedPdfFile, false);
      setSelectedPdfFile(null);
      if (pdfFileInputRef.current) pdfFileInputRef.current.value = "";
    } else if (!selectedPdfFile) {
      alert("Please select a PDF file to upload.");
    }
  };

  return (
    <aside className="w-72 border-r bg-muted/30 flex flex-col h-full">
      {/* Top Actions */}
      <div className="p-4 space-y-2">
        <Button
          onClick={onNewChat}
          className="w-full justify-start gap-2"
          variant="default"
        >
          <MessageSquarePlus className="h-4 w-4" />
          New Chat
        </Button>
        <Button
          onClick={onViewDocuments}
          className="w-full justify-start gap-2"
          variant="outline"
        >
          <FileText className="h-4 w-4" />
          View Documents
        </Button>
      </div>

      <Separator />

      {/* Conversations List */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="px-4 py-3 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Conversations
            </h2>
            {isInitialized && sessionIds.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleSelectionMode}
                className="h-7 text-xs"
              >
                {isSelectionMode ? (
                  <>
                    <X className="h-3 w-3 mr-1" />
                    Cancel
                  </>
                ) : (
                  <>
                    <CheckSquare className="h-3 w-3 mr-1" />
                    Select
                  </>
                )}
              </Button>
            )}
          </div>

          {/* Selection controls */}
          {isSelectionMode && (
            <div className="flex items-center justify-between gap-2">
              <div className="flex gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={selectAll}
                  className="h-7 text-xs"
                  disabled={selectedSessions.size === sessionIds.length}
                >
                  Select All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={deselectAll}
                  className="h-7 text-xs"
                  disabled={selectedSessions.size === 0}
                >
                  Clear
                </Button>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleBulkDelete}
                className="h-7 text-xs"
                disabled={selectedSessions.size === 0}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Delete ({selectedSessions.size})
              </Button>
            </div>
          )}
        </div>

        <ScrollArea className="flex-1 px-2 scrollbar-thin">
          {!isInitialized && (
            <div className="px-2 py-4 text-sm text-muted-foreground text-center">
              Loading...
            </div>
          )}
          {isInitialized && sessionIds.length === 0 && (
            <div className="px-2 py-8 text-sm text-muted-foreground text-center">
              No chats yet. Create one to get started!
            </div>
          )}

          <div className="space-y-1 pb-4">
            {isInitialized &&
              sessionIds.map((sessionId) => {
                const session = sessions[sessionId];
                const displayName =
                  session?.name || formatSessionIdFallback(sessionId);
                const isEditing = editingSessionId === sessionId;
                const isActive = sessionId === activeSessionId;

                return (
                  <div
                    key={sessionId}
                    className={cn(
                      "group relative rounded-md transition-colors",
                      isActive && "bg-accent"
                    )}
                  >
                    {isEditing ? (
                      <div className="flex items-center gap-1 p-2">
                        <Input
                          ref={editInputRef}
                          type="text"
                          value={editingValue}
                          onChange={handleInputChange}
                          onKeyDown={handleInputKeyDown}
                          className="h-8 text-sm"
                          aria-label={`Rename chat ${displayName}`}
                        />
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-8 w-8"
                          onClick={handleSaveEdit}
                        >
                          <Check className="h-4 w-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="h-8 w-8"
                          onClick={handleCancelEdit}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 pr-2">
                        {/* Checkbox in selection mode */}
                        {isSelectionMode && (
                          <Checkbox
                            checked={selectedSessions.has(sessionId)}
                            onCheckedChange={() => toggleSessionSelection(sessionId)}
                            className="ml-2"
                          />
                        )}

                        <button
                          onClick={() => isSelectionMode ? toggleSessionSelection(sessionId) : onSelectSession(sessionId)}
                          className="flex-1 text-left px-3 py-2 text-sm truncate hover:bg-accent/50 rounded-md transition-colors"
                          aria-current={isActive ? "page" : undefined}
                          title={displayName}
                        >
                          {displayName}
                        </button>

                        {/* Action buttons - hide in selection mode */}
                        {!isSelectionMode && (
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-7 w-7"
                              onClick={(e) => handleEditClick(e, sessionId)}
                              aria-label={`Rename ${displayName}`}
                              title="Rename"
                            >
                              <Pencil className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                              onClick={(e) => {
                                e.stopPropagation();
                                onDeleteSession(sessionId);
                              }}
                              aria-label={`Delete ${displayName}`}
                              title="Delete"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </ScrollArea>
      </div>

      <Separator />

      {/* Bottom Sections */}
      <div className="p-4 space-y-4 overflow-y-auto scrollbar-thin max-h-[400px]">
        {/* PDF Upload Section */}
        <Card>
          <CardHeader className="p-4 pb-3">
            <CardTitle className="text-sm">Upload Document</CardTitle>
            <CardDescription className="text-xs">
              Upload a PDF file to the knowledge base
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-0 space-y-2">
            <input
              type="file"
              ref={pdfFileInputRef}
              onChange={handlePdfFileChange}
              accept=".pdf,application/pdf"
              className="hidden"
              id="pdf-upload-input"
            />
            <label htmlFor="pdf-upload-input">
              <div className="flex items-center justify-center w-full h-20 border-2 border-dashed rounded-md cursor-pointer hover:border-primary transition-colors">
                <div className="text-center">
                  <FileUp className="h-6 w-6 mx-auto mb-1 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">
                    {selectedPdfFile
                      ? selectedPdfFile.name.substring(0, 25) +
                        (selectedPdfFile.name.length > 25 ? "..." : "")
                      : "Click to select PDF"}
                  </p>
                </div>
              </div>
            </label>
            <Button
              onClick={handlePdfUploadClick}
              disabled={isUploadingPdf || !selectedPdfFile}
              className="w-full"
              size="sm"
            >
              {isUploadingPdf ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload PDF
                </>
              )}
            </Button>
            {pdfUploadStatus && (
              <p
                className={cn(
                  "text-xs",
                  pdfUploadStatus.success
                    ? "text-primary"
                    : "text-destructive"
                )}
              >
                {pdfUploadStatus.message}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Automation Section */}
        <Card>
          <CardHeader className="p-4 pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Automate</CardTitle>
              <input
                type="file"
                ref={automationFileInputRef}
                onChange={handleJsonFileChange}
                accept=".json,application/json"
                className="hidden"
              />
              <Button
                size="sm"
                variant="ghost"
                onClick={handleUploadJsonClick}
                disabled={isSubmitting}
                className="h-7 text-xs"
              >
                <Upload className="h-3 w-3 mr-1" />
                Upload JSON
              </Button>
            </div>
            <CardDescription className="text-xs">
              Run automated conversation sequences
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4 pt-0 space-y-2">
            <textarea
              className={cn(
                "w-full px-3 py-2 text-xs font-mono rounded-md border border-input bg-background",
                "focus:outline-none focus:ring-2 focus:ring-ring",
                "disabled:cursor-not-allowed disabled:opacity-50",
                "resize-none"
              )}
              value={automationJson}
              onChange={(e) => setAutomationJson(e.target.value)}
              rows={4}
              placeholder='{ "inputs": ["Hello!", "Tell me a joke."] }'
              disabled={isSubmitting}
            />
            {automationError && (
              <p className="text-xs text-destructive" role="alert">
                {automationError}
              </p>
            )}
            <Button
              onClick={handleAutomationSubmit}
              disabled={
                isSubmitting || !activeSessionId || !selectedModel || !isInitialized
              }
              className="w-full"
              size="sm"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Automation
                </>
              )}
            </Button>
            {!selectedModel && activeSessionId && isInitialized && (
              <p className="text-xs text-muted-foreground">
                Select a model above
              </p>
            )}
            {!activeSessionId && isInitialized && (
              <p className="text-xs text-muted-foreground">
                Create or select a chat
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </aside>
  );
}

Sidebar.propTypes = {
  sessions: PropTypes.object.isRequired,
  activeSessionId: PropTypes.string,
  selectedModel: PropTypes.string,
  onNewChat: PropTypes.func.isRequired,
  onSelectSession: PropTypes.func.isRequired,
  onDeleteSession: PropTypes.func.isRequired,
  onRenameSession: PropTypes.func.isRequired,
  onAutomateConversation: PropTypes.func.isRequired,
  isSubmitting: PropTypes.bool,
  automationError: PropTypes.string,
  isInitialized: PropTypes.bool.isRequired,
  onUploadPdf: PropTypes.func,
  isUploadingPdf: PropTypes.bool,
  pdfUploadStatus: PropTypes.object,
  onViewDocuments: PropTypes.func,
};

export default Sidebar;
