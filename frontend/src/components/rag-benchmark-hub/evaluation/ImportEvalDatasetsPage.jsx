import { useState, useEffect, useRef } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { Separator } from "../../ui/separator";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Download,
  ExternalLink,
  FileText,
  Loader2,
  Upload,
  ArrowRight,
} from "lucide-react";
import { cn } from "../../../lib/utils";
import { API_BASE_URL } from "../../../lib/api-config";

export function ImportEvalDatasetsPage({ onImportComplete }) {
  // HuggingFace datasets state
  const [hfDatasets, setHfDatasets] = useState([]);
  const [hfImportStates, setHfImportStates] = useState({});

  // PDF datasets state
  const [pdfDatasets, setPdfDatasets] = useState([]);
  const [pdfImportStates, setPdfImportStates] = useState({});

  // Custom PDF upload state
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available datasets on mount
  useEffect(() => {
    fetchAllDatasets();
  }, []);

  const fetchAllDatasets = async () => {
    try {
      const [hfResponse, pdfResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/evaluation/import/datasets`),
        fetch(`${API_BASE_URL}/api/evaluation/import/pdf-datasets`),
      ]);

      if (hfResponse.ok) {
        const hfData = await hfResponse.json();
        setHfDatasets(hfData);
        const initialStates = {};
        hfData.forEach(ds => {
          initialStates[ds.id] = { rowCount: 500, customName: "", importing: false, completed: false, error: null, result: null };
        });
        setHfImportStates(initialStates);
      }

      if (pdfResponse.ok) {
        const pdfData = await pdfResponse.json();
        setPdfDatasets(pdfData);
        const initialStates = {};
        pdfData.forEach(ds => {
          initialStates[ds.id] = { customName: "", importing: false, completed: false, error: null, result: null };
        });
        setPdfImportStates(initialStates);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // HuggingFace import handlers
  const handleHfRowCountChange = (datasetId, value) => {
    const numValue = parseInt(value) || 0;
    setHfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], rowCount: Math.max(1, Math.min(10000, numValue)) }
    }));
  };

  const handleHfNameChange = (datasetId, value) => {
    setHfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], customName: value }
    }));
  };

  // HuggingFace import as raw dataset
  const handleHfImport = async (datasetId) => {
    const state = hfImportStates[datasetId];
    setHfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], importing: true, error: null, completed: false }
    }));

    try {
      const customName = state.customName?.trim() || "";
      let url = `${API_BASE_URL}/api/evaluation/import/huggingface-as-raw?dataset_id=${encodeURIComponent(datasetId)}&limit=${state.rowCount}`;
      if (customName) {
        url += `&custom_name=${encodeURIComponent(customName)}`;
      }
      const response = await fetch(url, { method: "POST" });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Import failed");
      }

      const result = await response.json();
      setHfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, completed: true, result }
      }));
    } catch (err) {
      setHfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, error: err.message }
      }));
    }
  };

  // PDF import handlers - now just imports as raw dataset
  const handlePdfNameChange = (datasetId, value) => {
    setPdfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], customName: value }
    }));
  };

  const handlePdfImport = async (datasetId) => {
    const dataset = pdfDatasets.find(d => d.id === datasetId);
    const state = pdfImportStates[datasetId];
    if (!dataset) return;

    // Check if this requires manual download
    if (dataset.requires_manual_download) {
      setPdfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], error: "This PDF requires manual download. Please download from the link and use 'Upload Custom PDF' below." }
      }));
      return;
    }

    setPdfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], importing: true, error: null, completed: false }
    }));

    try {
      const datasetName = state.customName?.trim() || dataset.name;
      const endpoint = `${API_BASE_URL}/api/evaluation/import/pdf-as-raw?dataset_id=${encodeURIComponent(datasetId)}&custom_name=${encodeURIComponent(datasetName)}`;

      const response = await fetch(endpoint, { method: "POST" });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Import failed");
      }

      const result = await response.json();
      setPdfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, completed: true, result }
      }));
    } catch (err) {
      setPdfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, error: err.message }
      }));
    }
  };

  // Custom PDF upload handlers
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setUploadFile(file);
      setUploadName(file.name.replace(".pdf", ""));
      setUploadStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;

    setUploadStatus({ status: "uploading", message: "Uploading..." });

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("dataset_name", uploadName);

      const response = await fetch(`${API_BASE_URL}/api/evaluation/import/pdf-upload-as-raw`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Upload failed");
      }

      const result = await response.json();
      setUploadStatus({ status: "completed", result });
    } catch (err) {
      setUploadStatus({ status: "failed", error: err.message });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-destructive p-4 bg-destructive/10 rounded-md">
        <AlertCircle className="h-5 w-5" />
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Import Raw Datasets</h2>
        <p className="text-muted-foreground">
          Import documents from HuggingFace or PDFs as raw datasets for preprocessing and Q&A generation
        </p>
      </div>

      {/* Workflow explanation */}
      <Card className="bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-4">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">Workflow:</span>
            <span className="text-muted-foreground">Import Raw</span>
            <ArrowRight className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Preprocess</span>
            <ArrowRight className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Generate Q&A</span>
            <ArrowRight className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Run Evaluation</span>
          </div>
        </CardContent>
      </Card>

      {/* HuggingFace Datasets Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          <h3 className="text-lg font-semibold">HuggingFace Datasets</h3>
        </div>
        <p className="text-sm text-muted-foreground">
          Import documents from HuggingFace as raw datasets for preprocessing and Q&A generation.
        </p>

        <div className="grid gap-4">
          {hfDatasets.map((dataset) => {
            const state = hfImportStates[dataset.id] || {};
            return (
              <Card key={dataset.id} className={cn(
                "transition-all",
                state.completed && "border-green-500/50 bg-green-50/50 dark:bg-green-950/20"
              )}>
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-base">{dataset.name}</CardTitle>
                      <CardDescription>{dataset.description}</CardDescription>
                    </div>
                    <a href={`https://huggingface.co/datasets/${dataset.id}`} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary">
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-end gap-4 flex-wrap">
                    <div className="flex-1 min-w-[200px]">
                      <Label className="text-xs">Dataset Name (optional)</Label>
                      <Input
                        type="text"
                        placeholder={dataset.name}
                        value={state.customName || ""}
                        onChange={(e) => handleHfNameChange(dataset.id, e.target.value)}
                        disabled={state.importing || state.completed}
                        className="mt-1 h-9"
                      />
                    </div>
                    <div className="w-[120px]">
                      <Label className="text-xs">Max items</Label>
                      <Input type="number" min={1} max={10000} value={state.rowCount || 500} onChange={(e) => handleHfRowCountChange(dataset.id, e.target.value)} disabled={state.importing || state.completed} className="mt-1 h-9" />
                    </div>
                    <Button onClick={() => handleHfImport(dataset.id)} disabled={state.importing || state.completed} variant={state.completed ? "outline" : "default"} size="sm">
                      {state.importing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Importing...</> : state.completed ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Imported</> : <><Download className="h-4 w-4 mr-2" />Import as Raw</>}
                    </Button>
                  </div>
                  {state.completed && state.result && (
                    <div className="p-3 bg-green-50 dark:bg-green-950/30 rounded-md">
                      <p className="text-sm text-green-700 dark:text-green-300">
                        <CheckCircle2 className="h-4 w-4 inline mr-1" />
                        Raw dataset created with {state.result.document_count} documents! Go to <strong>Data Management</strong> to preprocess it.
                      </p>
                    </div>
                  )}
                  {state.error && <div className="mt-2 text-destructive text-sm flex items-center gap-1"><AlertCircle className="h-3 w-3" />{state.error}</div>}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      <Separator />

      {/* PDF Datasets Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          <h3 className="text-lg font-semibold">PDF Documents (Romanian)</h3>
        </div>
        <p className="text-sm text-muted-foreground">
          Import bundled PDF documents as raw datasets.
        </p>

        <div className="grid gap-4">
          {pdfDatasets.map((dataset) => {
            const state = pdfImportStates[dataset.id] || {};
            const requiresManualDownload = dataset.requires_manual_download;
            return (
              <Card key={dataset.id} className={cn(
                "transition-all",
                state.completed && "border-green-500/50 bg-green-50/50 dark:bg-green-950/20",
                requiresManualDownload && "opacity-75"
              )}>
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <CardTitle className="text-base">{dataset.name}</CardTitle>
                        {dataset.local_file && (
                          <span className="text-xs bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 px-2 py-0.5 rounded">Bundled</span>
                        )}
                        {requiresManualDownload && (
                          <span className="text-xs bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300 px-2 py-0.5 rounded">Manual Download</span>
                        )}
                      </div>
                      <CardDescription>
                        {dataset.description}
                        {dataset.pages && <span className="ml-1">· ~{dataset.pages} pages</span>}
                      </CardDescription>
                    </div>
                    <a href={dataset.url} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary" title="View source">
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </div>
                </CardHeader>
                <CardContent>
                  {requiresManualDownload ? (
                    <p className="text-sm text-muted-foreground">
                      This PDF requires manual download due to website restrictions. Click the link above to download, then use "Upload Custom PDF" below.
                    </p>
                  ) : (
                    <>
                      <div className="flex items-end gap-4 flex-wrap">
                        <div className="flex-1 min-w-[200px]">
                          <Label className="text-xs">Dataset Name (optional)</Label>
                          <Input
                            type="text"
                            placeholder={dataset.name}
                            value={state.customName || ""}
                            onChange={(e) => handlePdfNameChange(dataset.id, e.target.value)}
                            disabled={state.importing || state.completed}
                            className="mt-1 h-9"
                          />
                        </div>
                        <Button onClick={() => handlePdfImport(dataset.id)} disabled={state.importing || state.completed} variant={state.completed ? "outline" : "default"} size="sm">
                          {state.importing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Importing...</> : state.completed ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Imported</> : <><Download className="h-4 w-4 mr-2" />Import as Raw</>}
                        </Button>
                      </div>
                      {state.completed && state.result && (
                        <div className="mt-3 p-3 bg-green-50 dark:bg-green-950/30 rounded-md">
                          <p className="text-sm text-green-700 dark:text-green-300">
                            <CheckCircle2 className="h-4 w-4 inline mr-1" />
                            Raw dataset created! Go to <strong>Data Management → Raw Datasets</strong> to preprocess it.
                          </p>
                        </div>
                      )}
                    </>
                  )}
                  {state.error && <div className="mt-2 text-destructive text-sm flex items-center gap-1"><AlertCircle className="h-3 w-3" />{state.error}</div>}
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Custom PDF Upload */}
        <Card className="border-dashed">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Custom PDF
            </CardTitle>
            <CardDescription>Upload your own PDF as a raw dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept=".pdf" className="hidden" />
              <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={uploadStatus?.status === "uploading"}>
                <Upload className="h-4 w-4 mr-2" />
                {uploadFile ? uploadFile.name : "Select PDF"}
              </Button>
              {uploadFile && (
                <span className="text-sm text-muted-foreground">
                  {(uploadFile.size / 1024 / 1024).toFixed(1)} MB
                </span>
              )}
            </div>

            {uploadFile && (
              <div className="flex items-end gap-4 flex-wrap">
                <div className="flex-1 min-w-[200px]">
                  <Label className="text-xs">Dataset Name</Label>
                  <Input value={uploadName} onChange={(e) => setUploadName(e.target.value)} placeholder="My Dataset" className="mt-1 h-9" disabled={uploadStatus?.status === "uploading"} />
                </div>
                <Button onClick={handleUpload} disabled={!uploadName || uploadStatus?.status === "uploading" || uploadStatus?.status === "completed"}>
                  {uploadStatus?.status === "uploading" ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Uploading...</> : uploadStatus?.status === "completed" ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Imported</> : "Import as Raw"}
                </Button>
              </div>
            )}

            {uploadStatus?.error && (
              <div className="text-destructive text-sm flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                {uploadStatus.error}
              </div>
            )}

            {uploadStatus?.status === "completed" && (
              <div className="p-3 bg-green-50 dark:bg-green-950/30 rounded-md">
                <p className="text-sm text-green-700 dark:text-green-300">
                  <CheckCircle2 className="h-4 w-4 inline mr-1" />
                  Raw dataset "{uploadStatus.result?.name}" created! Go to <strong>Data Management → Raw Datasets</strong> to preprocess it.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tips */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground">
            <strong>Tip:</strong> After importing a PDF, preprocess it to create searchable chunks.
            Then you can generate Q&A pairs from the processed content to use for evaluation.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default ImportEvalDatasetsPage;
