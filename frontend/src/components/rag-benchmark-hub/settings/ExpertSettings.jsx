import { useState } from "react";
import {
  AlertTriangle,
  Database,
  FileX2,
  FlaskConical,
  HardDrive,
  Loader2,
  Settings,
  Shield,
  Trash2,
} from "lucide-react";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import { Badge } from "../../ui/badge";
import { Separator } from "../../ui/separator";
import { Checkbox } from "../../ui/checkbox";
import { API_BASE_URL } from "../../../lib/api-config";

const CONFIRMATION_PHRASE = "DELETE ALL DATA";

export function ExpertSettings() {
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [confirmationInput, setConfirmationInput] = useState("");
  const [isResetting, setIsResetting] = useState(false);
  const [resetOptions, setResetOptions] = useState({
    evaluationRuns: true,
    evaluationDatasets: true,
    evaluationTasks: true,
    processedDatasets: false,
    rawDatasets: false,
    vectorCollections: false,
  });
  const [resetResult, setResetResult] = useState(null);

  const isConfirmationValid = confirmationInput === CONFIRMATION_PHRASE;

  const selectedCount = Object.values(resetOptions).filter(Boolean).length;

  const handleReset = async () => {
    if (!isConfirmationValid) return;

    setIsResetting(true);
    setResetResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/admin/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          evaluation_runs: resetOptions.evaluationRuns,
          evaluation_datasets: resetOptions.evaluationDatasets,
          evaluation_tasks: resetOptions.evaluationTasks,
          processed_datasets: resetOptions.processedDatasets,
          raw_datasets: resetOptions.rawDatasets,
          vector_collections: resetOptions.vectorCollections,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setResetResult({ success: true, ...result });
        setShowResetDialog(false);
        setConfirmationInput("");
      } else {
        const error = await response.json();
        setResetResult({ success: false, error: error.detail || "Reset failed" });
      }
    } catch (err) {
      setResetResult({ success: false, error: err.message });
    } finally {
      setIsResetting(false);
    }
  };

  const resetOptionsList = [
    {
      key: "evaluationRuns",
      label: "Evaluation Runs",
      description: "Delete all saved evaluation results and metrics",
      icon: FlaskConical,
      danger: "medium",
    },
    {
      key: "evaluationDatasets",
      label: "Evaluation Datasets",
      description: "Delete all Q&A test datasets",
      icon: FileX2,
      danger: "medium",
    },
    {
      key: "evaluationTasks",
      label: "Evaluation Tasks",
      description: "Clear evaluation task history from database",
      icon: Database,
      danger: "low",
    },
    {
      key: "processedDatasets",
      label: "Processed Datasets",
      description: "Delete all processed/chunked datasets",
      icon: HardDrive,
      danger: "high",
    },
    {
      key: "rawDatasets",
      label: "Raw Datasets",
      description: "Delete all uploaded raw documents",
      icon: FileX2,
      danger: "high",
    },
    {
      key: "vectorCollections",
      label: "Vector Collections",
      description: "Delete all vector embeddings from Qdrant",
      icon: Database,
      danger: "high",
    },
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Settings className="h-6 w-6" />
          Expert Settings
        </h2>
        <p className="text-muted-foreground">
          Advanced options for managing your benchmark data. Use with caution.
        </p>
      </div>

      {/* Warning Banner */}
      <Card className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20">
        <CardContent className="flex items-start gap-3 p-4">
          <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-yellow-800 dark:text-yellow-200">
              Danger Zone
            </p>
            <p className="text-sm text-yellow-700 dark:text-yellow-300">
              Actions on this page can permanently delete data. Make sure you have backups before proceeding.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Success/Error Result */}
      {resetResult && (
        <Card className={resetResult.success ? "border-green-500 bg-green-50 dark:bg-green-950/20" : "border-red-500 bg-red-50 dark:bg-red-950/20"}>
          <CardContent className="p-4">
            {resetResult.success ? (
              <div>
                <p className="font-medium text-green-800 dark:text-green-200 mb-2">
                  Reset completed successfully
                </p>
                <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
                  {resetResult.deleted_evaluation_runs > 0 && (
                    <li>• Deleted {resetResult.deleted_evaluation_runs} evaluation runs</li>
                  )}
                  {resetResult.deleted_evaluation_datasets > 0 && (
                    <li>• Deleted {resetResult.deleted_evaluation_datasets} evaluation datasets</li>
                  )}
                  {resetResult.deleted_evaluation_tasks > 0 && (
                    <li>• Deleted {resetResult.deleted_evaluation_tasks} evaluation tasks</li>
                  )}
                  {resetResult.deleted_processed_datasets > 0 && (
                    <li>• Deleted {resetResult.deleted_processed_datasets} processed datasets</li>
                  )}
                  {resetResult.deleted_raw_datasets > 0 && (
                    <li>• Deleted {resetResult.deleted_raw_datasets} raw datasets</li>
                  )}
                  {resetResult.deleted_vector_collections > 0 && (
                    <li>• Deleted {resetResult.deleted_vector_collections} vector collections</li>
                  )}
                </ul>
              </div>
            ) : (
              <p className="text-red-800 dark:text-red-200">
                Error: {resetResult.error}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Data Reset Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <Trash2 className="h-5 w-5" />
            Data Reset
          </CardTitle>
          <CardDescription>
            Selectively delete benchmark data to start fresh. This action cannot be undone.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3">
            {resetOptionsList.map((option) => {
              const Icon = option.icon;
              return (
                <div
                  key={option.key}
                  className={`flex items-start gap-3 p-3 rounded-lg border ${
                    resetOptions[option.key]
                      ? option.danger === "high"
                        ? "border-red-300 bg-red-50 dark:bg-red-950/20"
                        : "border-yellow-300 bg-yellow-50 dark:bg-yellow-950/20"
                      : "border-muted"
                  }`}
                >
                  <Checkbox
                    id={option.key}
                    checked={resetOptions[option.key]}
                    onCheckedChange={(checked) =>
                      setResetOptions((prev) => ({ ...prev, [option.key]: checked }))
                    }
                  />
                  <div className="flex-1">
                    <label
                      htmlFor={option.key}
                      className="flex items-center gap-2 font-medium cursor-pointer"
                    >
                      <Icon className="h-4 w-4" />
                      {option.label}
                      {option.danger === "high" && (
                        <Badge variant="destructive" className="text-xs">
                          High Impact
                        </Badge>
                      )}
                    </label>
                    <p className="text-sm text-muted-foreground mt-0.5">
                      {option.description}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                {selectedCount} item{selectedCount !== 1 ? "s" : ""} selected for deletion
              </p>
            </div>
            <Button
              variant="destructive"
              onClick={() => setShowResetDialog(true)}
              disabled={selectedCount === 0}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Reset Selected Data
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Confirmation Dialog */}
      <Dialog open={showResetDialog} onOpenChange={setShowResetDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-600">
              <Shield className="h-5 w-5" />
              Confirm Data Deletion
            </DialogTitle>
            <DialogDescription>
              This action will permanently delete the selected data. This cannot be undone.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Summary of what will be deleted */}
            <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 rounded-lg p-3">
              <p className="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
                The following will be permanently deleted:
              </p>
              <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
                {resetOptions.evaluationRuns && <li>• All evaluation runs and results</li>}
                {resetOptions.evaluationDatasets && <li>• All Q&A evaluation datasets</li>}
                {resetOptions.evaluationTasks && <li>• All evaluation task records</li>}
                {resetOptions.processedDatasets && <li>• All processed datasets and chunks</li>}
                {resetOptions.rawDatasets && <li>• All raw uploaded documents</li>}
                {resetOptions.vectorCollections && <li>• All vector embeddings in Qdrant</li>}
              </ul>
            </div>

            {/* Type to confirm */}
            <div className="space-y-2">
              <Label htmlFor="confirmation">
                Type <span className="font-mono font-bold text-red-600">{CONFIRMATION_PHRASE}</span> to confirm:
              </Label>
              <Input
                id="confirmation"
                value={confirmationInput}
                onChange={(e) => setConfirmationInput(e.target.value)}
                placeholder="Type confirmation phrase..."
                className={
                  confirmationInput && !isConfirmationValid
                    ? "border-red-500 focus-visible:ring-red-500"
                    : ""
                }
              />
              {confirmationInput && !isConfirmationValid && (
                <p className="text-xs text-red-500">
                  Phrase doesn't match. Please type exactly: {CONFIRMATION_PHRASE}
                </p>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowResetDialog(false);
                setConfirmationInput("");
              }}
              disabled={isResetting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReset}
              disabled={!isConfirmationValid || isResetting}
            >
              {isResetting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Forever
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default ExpertSettings;
