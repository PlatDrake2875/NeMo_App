import PropTypes from "prop-types";
import { useState, useCallback } from "react";
import { FileText, Upload, AlertCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Alert } from "./ui/alert";
import { Badge } from "./ui/badge";
import { parseImportedConversation } from "../utils/exportUtils";

/**
 * ImportConversationDialog - Dialog for importing JSON conversations
 * @param {Object} props
 * @param {boolean} props.open - Whether the dialog is open
 * @param {function} props.onOpenChange - Handler for open state changes
 * @param {function} props.onImport - Handler for importing (receives parsed data)
 */
export function ImportConversationDialog({
  open,
  onOpenChange,
  onImport
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [parsedData, setParsedData] = useState(null);
  const [error, setError] = useState(null);
  const [isImporting, setIsImporting] = useState(false);

  const resetState = useCallback(() => {
    setSelectedFile(null);
    setParsedData(null);
    setError(null);
    setIsImporting(false);
  }, []);

  const handleOpenChange = useCallback((newOpen) => {
    if (!newOpen) {
      resetState();
    }
    onOpenChange(newOpen);
  }, [onOpenChange, resetState]);

  const handleFileSelect = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setError(null);
    setParsedData(null);

    try {
      const text = await file.text();
      const parsed = parseImportedConversation(text);
      setParsedData(parsed);
    } catch (err) {
      console.error("Failed to parse file:", err);
      setError(err.message || "Failed to parse file. Please ensure it's a valid conversation JSON.");
    }
  }, []);

  const handleImport = useCallback(async () => {
    if (!parsedData) return;

    setIsImporting(true);
    try {
      await onImport(parsedData);
      handleOpenChange(false);
    } catch (err) {
      console.error("Import failed:", err);
      setError(err.message || "Failed to import conversation.");
    } finally {
      setIsImporting(false);
    }
  }, [parsedData, onImport, handleOpenChange]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Import Conversation</DialogTitle>
          <DialogDescription>
            Import a previously exported conversation from a JSON file.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-4">
          {/* File Upload Area */}
          <div
            className={`
              border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer
              ${parsedData ? "border-green-500 bg-green-50 dark:bg-green-900/10" : "border-muted-foreground/25 hover:border-primary/50"}
            `}
            onClick={() => document.getElementById("import-file-input")?.click()}
          >
            <input
              id="import-file-input"
              type="file"
              accept=".json"
              onChange={handleFileSelect}
              className="hidden"
            />
            {parsedData ? (
              <>
                <FileText className="h-10 w-10 mx-auto text-green-500 mb-3" />
                <p className="text-sm font-medium">{selectedFile?.name}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Click to select a different file
                </p>
              </>
            ) : (
              <>
                <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                <p className="text-sm text-muted-foreground">
                  Click to select a conversation JSON file
                </p>
              </>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <span className="ml-2 text-sm">{error}</span>
            </Alert>
          )}

          {/* Preview */}
          {parsedData && (
            <div className="p-4 bg-muted rounded-lg space-y-3">
              <h4 className="font-medium text-sm">Preview</h4>
              <div className="space-y-2 text-sm">
                {parsedData.sessionInfo?.name && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Session Name:</span>
                    <span>{parsedData.sessionInfo.name}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Messages:</span>
                  <Badge variant="secondary">{parsedData.messages.length}</Badge>
                </div>
                {parsedData.sessionInfo?.originalExportDate && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Originally Exported:</span>
                    <span className="text-xs">
                      {new Date(parsedData.sessionInfo.originalExportDate).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>

              {/* Message Preview */}
              <div className="mt-3 pt-3 border-t">
                <p className="text-xs text-muted-foreground mb-2">First message:</p>
                <p className="text-xs line-clamp-2 bg-background p-2 rounded">
                  {parsedData.messages[0]?.text || "No messages"}
                </p>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={isImporting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            disabled={!parsedData || isImporting}
          >
            {isImporting ? "Importing..." : "Import as New Chat"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

ImportConversationDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onOpenChange: PropTypes.func.isRequired,
  onImport: PropTypes.func.isRequired
};
