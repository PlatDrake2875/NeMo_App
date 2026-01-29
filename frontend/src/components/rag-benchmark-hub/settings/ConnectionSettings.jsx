import { useState, useCallback, useEffect } from "react";
import {
  Globe,
  CheckCircle2,
  XCircle,
  Loader2,
  RotateCcw,
  Save,
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
import { Badge } from "../../ui/badge";
import {
  getApiBaseUrl,
  setApiBaseUrl,
  getDefaultUrls,
} from "../../../lib/api-config";

export function ConnectionSettings() {
  const defaults = getDefaultUrls();

  // Local state for form input
  const [localBackendUrl, setLocalBackendUrl] = useState(getApiBaseUrl());
  const [storedBackendUrl, setStoredBackendUrl] = useState(getApiBaseUrl());

  // Connection test state
  const [backendStatus, setBackendStatus] = useState("unknown");
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState(null);

  // Test backend connection on mount (but don't show error for default localhost)
  useEffect(() => {
    testBackendConnection(storedBackendUrl).catch(() => {
      // Silently fail on mount - user will see "Not tested" or "Failed" badge
    });
  }, []);

  const testBackendConnection = useCallback(async (url) => {
    setBackendStatus("checking");
    setSaveMessage(null);
    try {
      const response = await fetch(`${url}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) {
        setBackendStatus("connected");
        return true;
      }
      setBackendStatus("error");
      return false;
    } catch (e) {
      console.error("Backend connection test failed:", e);
      setBackendStatus("error");
      // Don't show error message for localhost - it's expected if backend is remote
      if (!url.includes("localhost")) {
        setSaveMessage({
          type: "error",
          text: `Cannot connect to ${url}. Check the URL and ensure the backend is running.`,
        });
      }
      return false;
    }
  }, []);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    setSaveMessage(null);

    try {
      // Test backend connection first
      const backendOk = await testBackendConnection(localBackendUrl);
      if (!backendOk) {
        setSaveMessage({
          type: "error",
          text: "Could not connect to backend. Please check the URL.",
        });
        setIsSaving(false);
        return;
      }

      // Save to localStorage
      setApiBaseUrl(localBackendUrl);
      setStoredBackendUrl(localBackendUrl);

      setSaveMessage({
        type: "success",
        text: "Settings saved! Page will reload to apply changes.",
      });

      // Reload page after short delay to apply new backend URL everywhere
      setTimeout(() => {
        window.location.reload();
      }, 1500);
    } catch (e) {
      setSaveMessage({ type: "error", text: `Error saving settings: ${e.message}` });
    } finally {
      setIsSaving(false);
    }
  }, [localBackendUrl, testBackendConnection]);

  const handleReset = useCallback(() => {
    setLocalBackendUrl(defaults.backend);
    setBackendStatus("unknown");
    setSaveMessage(null);
  }, [defaults]);

  const handleClearAndReload = useCallback(() => {
    // Clear the stored URL and reload to use defaults
    localStorage.removeItem("nemo_backend_url");
    window.location.reload();
  }, []);

  const getStatusBadge = (status) => {
    switch (status) {
      case "connected":
        return (
          <Badge variant="default" className="gap-1 bg-green-600">
            <CheckCircle2 className="h-3 w-3" />
            Connected
          </Badge>
        );
      case "error":
        return (
          <Badge variant="destructive" className="gap-1">
            <XCircle className="h-3 w-3" />
            Failed
          </Badge>
        );
      case "checking":
        return (
          <Badge variant="secondary" className="gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            Testing...
          </Badge>
        );
      default:
        return <Badge variant="outline">Not tested</Badge>;
    }
  };

  const hasChanges = localBackendUrl !== storedBackendUrl;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Globe className="h-5 w-5" />
          Connection Settings
        </CardTitle>
        <CardDescription>
          Configure the backend API URL. Changes require a page reload.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Backend URL */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="backend-url" className="flex items-center gap-2">
              <Globe className="h-4 w-4" />
              Backend API URL
            </Label>
            {getStatusBadge(backendStatus)}
          </div>
          <div className="flex gap-2">
            <Input
              id="backend-url"
              type="url"
              value={localBackendUrl}
              onChange={(e) => setLocalBackendUrl(e.target.value)}
              placeholder="https://example.com or http://localhost:8000"
              className="font-mono text-sm"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => testBackendConnection(localBackendUrl)}
              disabled={backendStatus === "checking"}
            >
              Test
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            The URL where the NeMo backend is running (e.g., https://3f8852e85745.ratio1.link)
          </p>
        </div>

        {/* Save Message */}
        {saveMessage && (
          <div
            className={`p-3 rounded-lg ${
              saveMessage.type === "success"
                ? "bg-green-50 text-green-800 dark:bg-green-950 dark:text-green-200"
                : "bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200"
            }`}
          >
            {saveMessage.text}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleReset} disabled={isSaving}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset to Default
            </Button>
            <Button variant="ghost" size="sm" onClick={handleClearAndReload} className="text-muted-foreground">
              Clear & Reload
            </Button>
          </div>
          <Button onClick={handleSave} disabled={isSaving || !hasChanges}>
            {isSaving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save & Apply
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default ConnectionSettings;
