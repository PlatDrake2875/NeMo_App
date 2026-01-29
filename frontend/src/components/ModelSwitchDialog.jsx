/**
 * ModelSwitchDialog - Confirmation and progress dialog for model switching.
 *
 * Shows:
 * - Confirmation before starting switch
 * - Progress bar during switch
 * - Success/error states
 */

import { useState, useEffect, useRef } from "react";
import { Loader2, CheckCircle, XCircle, AlertTriangle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Alert, AlertDescription } from "./ui/alert";
import { getApiBaseUrl } from "../lib/api-config";

export function ModelSwitchDialog({
  open,
  onOpenChange,
  targetModel,
  currentModel,
  onConfirm,
  onComplete,
}) {
  // confirm | switching | success | error
  const [switchState, setSwitchState] = useState("confirm");
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("");
  const [error, setError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [activeTargetModel, setActiveTargetModel] = useState(null); // Store model being switched to
  const pollIntervalRef = useRef(null);

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setSwitchState("confirm");
      setProgress(0);
      setCurrentStep("");
      setError(null);
      setTaskId(null);
      setActiveTargetModel(targetModel); // Capture target model when dialog opens
    }
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [open, targetModel]);

  const startSwitch = async () => {
    setSwitchState("switching");
    setError(null);

    try {
      const response = await fetch(`${getApiBaseUrl()}/api/models/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: targetModel }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to start model switch");
      }

      const data = await response.json();
      setTaskId(data.id);
      setProgress(data.progress);
      setCurrentStep(data.current_step);

      // Start polling for progress
      pollIntervalRef.current = setInterval(() => pollProgress(data.id), 2000);
    } catch (err) {
      setError(err.message);
      setSwitchState("error");
    }
  };

  const pollProgress = async (id) => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/models/switch/${id}`);
      if (!response.ok) return;

      const data = await response.json();
      setProgress(data.progress);
      setCurrentStep(data.current_step);

      if (data.status === "ready") {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        setSwitchState("success");
        onComplete?.();
        // Auto-close after success
        setTimeout(() => onOpenChange(false), 2500);
      } else if (data.status === "failed" || data.status === "cancelled") {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        setError(data.error_message || "Model switch failed");
        setSwitchState("error");
      }
    } catch (err) {
      console.error("Error polling switch status:", err);
    }
  };

  const handleCancel = async () => {
    if (taskId && switchState === "switching") {
      try {
        await fetch(`${getApiBaseUrl()}/api/models/switch/${taskId}`, {
          method: "DELETE",
        });
      } catch (err) {
        console.error("Error cancelling switch:", err);
      }
    }
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    onOpenChange(false);
  };

  const getEstimatedTime = () => {
    if (progress < 30) return "~2-4 minutes remaining";
    if (progress < 60) return "~1-2 minutes remaining";
    if (progress < 90) return "< 1 minute remaining";
    return "Almost ready...";
  };

  const formatModelName = (model) => {
    if (!model) return "Unknown";
    return model.split("/").pop();
  };

  return (
    <Dialog open={open} onOpenChange={handleCancel}>
      <DialogContent className="sm:max-w-[450px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {switchState === "confirm" && "Switch Model"}
            {switchState === "switching" && (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Switching Model...
              </>
            )}
            {switchState === "success" && (
              <>
                <CheckCircle className="h-5 w-5 text-green-600" />
                Model Switched
              </>
            )}
            {switchState === "error" && (
              <>
                <XCircle className="h-5 w-5 text-destructive" />
                Switch Failed
              </>
            )}
          </DialogTitle>
          {switchState === "confirm" && (
            <DialogDescription>
              Switch from <strong>{formatModelName(currentModel)}</strong> to{" "}
              <strong>{formatModelName(activeTargetModel)}</strong>?
            </DialogDescription>
          )}
        </DialogHeader>

        <div className="py-4">
          {switchState === "confirm" && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                This will temporarily interrupt chat functionality while the new
                model loads (~1-5 minutes depending on model size).
              </AlertDescription>
            </Alert>
          )}

          {switchState === "switching" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">{currentStep || "Processing..."}</span>
              </div>
              <Progress value={progress} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{progress}%</span>
                <span>{getEstimatedTime()}</span>
              </div>
            </div>
          )}

          {switchState === "success" && (
            <Alert className="border-green-500/50 bg-green-500/10">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-700 dark:text-green-400">
                Model <strong>{formatModelName(activeTargetModel)}</strong> is now
                active and ready for use.
              </AlertDescription>
            </Alert>
          )}

          {switchState === "error" && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          {switchState === "confirm" && (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={startSwitch}>Switch Model</Button>
            </>
          )}

          {switchState === "switching" && (
            <Button variant="outline" onClick={handleCancel}>
              Cancel Switch
            </Button>
          )}

          {(switchState === "success" || switchState === "error") && (
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default ModelSwitchDialog;
