import { useState, useEffect, useRef } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Checkbox } from "../../ui/checkbox";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { Progress } from "../../ui/progress";
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
  const [uploadMaxPairs, setUploadMaxPairs] = useState(50);
  const [uploadAlsoCreateRaw, setUploadAlsoCreateRaw] = useState(true);
  const [uploadJobId, setUploadJobId] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available datasets on mount
  useEffect(() => {
    fetchAllDatasets();
  }, []);

  // Poll for PDF import status
  useEffect(() => {
    const activeJobs = [
      ...Object.entries(pdfImportStates).filter(([_, s]) => s.jobId && s.status !== "completed" && s.status !== "failed"),
      uploadJobId && uploadStatus && uploadStatus.status !== "completed" && uploadStatus.status !== "failed"
        ? [["upload", { jobId: uploadJobId }]]
        : []
    ].flat();

    if (activeJobs.length === 0) return;

    const interval = setInterval(async () => {
      for (const [id, state] of Object.entries(pdfImportStates)) {
        if (state.jobId && state.status !== "completed" && state.status !== "failed") {
          await pollPdfStatus(id, state.jobId, false);
        }
      }
      if (uploadJobId && uploadStatus && uploadStatus.status !== "completed" && uploadStatus.status !== "failed") {
        await pollPdfStatus("upload", uploadJobId, true);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [pdfImportStates, uploadJobId, uploadStatus]);

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
          initialStates[ds.id] = { rowCount: 500, importing: false, completed: false, error: null, result: null };
        });
        setHfImportStates(initialStates);
      }

      if (pdfResponse.ok) {
        const pdfData = await pdfResponse.json();
        setPdfDatasets(pdfData);
        const initialStates = {};
        pdfData.forEach(ds => {
          initialStates[ds.id] = { maxPairs: 50, customName: "", alsoCreateRaw: true, importing: false, completed: false, error: null, result: null, jobId: null, status: null, progress: 0 };
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

  const handleHfImport = async (datasetId) => {
    const state = hfImportStates[datasetId];
    setHfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], importing: true, error: null, completed: false }
    }));

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/evaluation/import/huggingface?dataset_id=${encodeURIComponent(datasetId)}&limit=${state.rowCount}`,
        { method: "POST" }
      );

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Import failed");
      }

      const result = await response.json();
      setHfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, completed: true, result }
      }));

      if (onImportComplete) onImportComplete(result);
    } catch (err) {
      setHfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, error: err.message }
      }));
    }
  };

  // PDF import handlers
  const handlePdfMaxPairsChange = (datasetId, value) => {
    const numValue = parseInt(value) || 0;
    setPdfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], maxPairs: Math.max(1, Math.min(500, numValue)) }
    }));
  };

  const handlePdfNameChange = (datasetId, value) => {
    setPdfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], customName: value }
    }));
  };

  const handleAlsoCreateRawChange = (datasetId, checked) => {
    setPdfImportStates(prev => ({
      ...prev,
      [datasetId]: { ...prev[datasetId], alsoCreateRaw: checked }
    }));
  };

  const pollPdfStatus = async (datasetId, jobId, isUpload = false) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/import/pdf-status/${jobId}`);
      if (response.ok) {
        const statusData = await response.json();

        if (isUpload) {
          setUploadStatus(statusData);
          if (statusData.status === "completed") {
            if (onImportComplete) onImportComplete({ id: statusData.dataset_id, pair_count: statusData.pairs_generated });
          }
        } else {
          setPdfImportStates(prev => ({
            ...prev,
            [datasetId]: {
              ...prev[datasetId],
              status: statusData.status,
              progress: statusData.progress,
              error: statusData.error,
              completed: statusData.status === "completed",
              importing: !["completed", "failed"].includes(statusData.status),
              result: statusData.status === "completed" ? { pair_count: statusData.pairs_generated, dataset_id: statusData.dataset_id } : null,
            }
          }));

          if (statusData.status === "completed" && onImportComplete) {
            onImportComplete({ id: statusData.dataset_id, pair_count: statusData.pairs_generated });
          }
        }
      }
    } catch (err) {
      console.error("Failed to poll PDF status:", err);
    }
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
      [datasetId]: { ...prev[datasetId], importing: true, error: null, completed: false, status: "starting", progress: 0 }
    }));

    try {
      // Use custom name if provided, otherwise use default dataset name
      const datasetName = state.customName?.trim() || dataset.name;
      const alsoCreateRaw = state.alsoCreateRaw !== false;  // Default true

      // Use local endpoint if bundled, otherwise URL endpoint
      const endpoint = dataset.local_file
        ? `${API_BASE_URL}/api/evaluation/import/pdf-local?dataset_id=${encodeURIComponent(datasetId)}&max_pairs=${state.maxPairs}&pairs_per_chunk=2&use_vllm=true&custom_name=${encodeURIComponent(datasetName)}&also_create_raw_dataset=${alsoCreateRaw}`
        : `${API_BASE_URL}/api/evaluation/import/pdf-url?url=${encodeURIComponent(dataset.url)}&dataset_name=${encodeURIComponent(datasetName)}&max_pairs=${state.maxPairs}&pairs_per_chunk=2&use_vllm=true&also_create_raw_dataset=${alsoCreateRaw}`;

      const response = await fetch(endpoint, { method: "POST" });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Import failed");
      }

      const result = await response.json();
      setPdfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], jobId: result.job_id, status: dataset.local_file ? "loading" : "downloading" }
      }));
    } catch (err) {
      setPdfImportStates(prev => ({
        ...prev,
        [datasetId]: { ...prev[datasetId], importing: false, error: err.message, status: "failed" }
      }));
    }
  };

  // Custom PDF upload handlers
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setUploadFile(file);
      setUploadName(file.name.replace(".pdf", ""));
    }
  };

  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;

    setUploadStatus({ status: "uploading", progress: 0, message: "Uploading..." });

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("dataset_name", uploadName);
      formData.append("max_pairs", uploadMaxPairs.toString());
      formData.append("pairs_per_chunk", "2");
      formData.append("use_vllm", "true");
      formData.append("also_create_raw_dataset", uploadAlsoCreateRaw.toString());

      const response = await fetch(`${API_BASE_URL}/api/evaluation/import/pdf-upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Upload failed");
      }

      const result = await response.json();
      setUploadJobId(result.job_id);
      setUploadStatus({ status: "processing", progress: 10, message: "Processing PDF..." });
    } catch (err) {
      setUploadStatus({ status: "failed", progress: 0, message: err.message, error: err.message });
    }
  };

  const getStatusMessage = (status) => {
    const messages = {
      downloading: "Downloading PDF...",
      extracting: "Extracting text...",
      chunking: "Chunking content...",
      generating: "Generating Q&A pairs...",
      completed: "Completed!",
      failed: "Failed",
    };
    return messages[status] || status;
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
        <h2 className="text-2xl font-bold tracking-tight">Import Evaluation Datasets</h2>
        <p className="text-muted-foreground">
          Import Q&A datasets from HuggingFace or generate from PDFs
        </p>
      </div>

      {/* HuggingFace Datasets Section */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          <h3 className="text-lg font-semibold">HuggingFace Datasets</h3>
        </div>

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
                <CardContent>
                  <div className="flex items-end gap-4">
                    <div className="flex-1 max-w-[180px]">
                      <Label className="text-xs">Q&A pairs</Label>
                      <Input type="number" min={1} max={10000} value={state.rowCount || 500} onChange={(e) => handleHfRowCountChange(dataset.id, e.target.value)} disabled={state.importing} className="mt-1 h-9" />
                    </div>
                    <Button onClick={() => handleHfImport(dataset.id)} disabled={state.importing || state.completed} variant={state.completed ? "outline" : "default"} size="sm">
                      {state.importing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Importing...</> : state.completed ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Done ({state.result?.pair_count})</> : <><Download className="h-4 w-4 mr-2" />Import</>}
                    </Button>
                  </div>
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
          Generate Q&A pairs from PDF documents using AI. Processing may take a few minutes.
        </p>

        <div className="grid gap-4">
          {pdfDatasets.map((dataset) => {
            const state = pdfImportStates[dataset.id] || {};
            const isProcessing = state.importing && !state.completed;
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
                    <a href={dataset.url} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary" title="Download PDF">
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
                            disabled={isProcessing || state.completed}
                            className="mt-1 h-9"
                          />
                        </div>
                        <div className="w-[120px]">
                          <Label className="text-xs">Max Q&A pairs</Label>
                          <Input type="number" min={10} max={500} value={state.maxPairs || 50} onChange={(e) => handlePdfMaxPairsChange(dataset.id, e.target.value)} disabled={isProcessing || state.completed} className="mt-1 h-9" />
                        </div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id={`raw-${dataset.id}`}
                            checked={state.alsoCreateRaw !== false}
                            onCheckedChange={(checked) => handleAlsoCreateRawChange(dataset.id, checked)}
                            disabled={isProcessing || state.completed}
                          />
                          <Label htmlFor={`raw-${dataset.id}`} className="text-xs cursor-pointer">
                            Also create raw dataset
                          </Label>
                        </div>
                        <Button onClick={() => handlePdfImport(dataset.id)} disabled={isProcessing || state.completed} variant={state.completed ? "outline" : "default"} size="sm">
                          {isProcessing ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />{getStatusMessage(state.status)}</> : state.completed ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Done ({state.result?.pair_count})</> : <><Download className="h-4 w-4 mr-2" />Generate</>}
                        </Button>
                      </div>
                      {isProcessing && state.progress > 0 && (
                        <div className="mt-3">
                          <Progress value={state.progress} className="h-2" />
                          <p className="text-xs text-muted-foreground mt-1">{state.progress}% - {getStatusMessage(state.status)}</p>
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
            <CardDescription>Upload your own PDF to generate Q&A pairs</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept=".pdf" className="hidden" />
              <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={uploadStatus?.status === "processing"}>
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
                  <Input value={uploadName} onChange={(e) => setUploadName(e.target.value)} placeholder="My Dataset" className="mt-1 h-9" disabled={uploadStatus?.status === "processing"} />
                </div>
                <div className="w-[120px]">
                  <Label className="text-xs">Max pairs</Label>
                  <Input type="number" min={10} max={500} value={uploadMaxPairs} onChange={(e) => setUploadMaxPairs(parseInt(e.target.value) || 50)} className="mt-1 h-9" disabled={uploadStatus?.status === "processing"} />
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="upload-raw"
                    checked={uploadAlsoCreateRaw}
                    onCheckedChange={(checked) => setUploadAlsoCreateRaw(checked)}
                    disabled={uploadStatus?.status === "processing" || uploadStatus?.status === "completed"}
                  />
                  <Label htmlFor="upload-raw" className="text-xs cursor-pointer">
                    Also create raw dataset
                  </Label>
                </div>
                <Button onClick={handleUpload} disabled={!uploadName || uploadStatus?.status === "processing" || uploadStatus?.status === "completed"}>
                  {uploadStatus?.status === "processing" ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Processing...</> : uploadStatus?.status === "completed" ? <><CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />Done ({uploadStatus.pairs_generated})</> : "Generate Q&A"}
                </Button>
              </div>
            )}

            {uploadStatus && uploadStatus.status !== "completed" && uploadStatus.progress > 0 && (
              <div>
                <Progress value={uploadStatus.progress} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">{uploadStatus.progress}% - {uploadStatus.message || getStatusMessage(uploadStatus.status)}</p>
              </div>
            )}

            {uploadStatus?.error && (
              <div className="text-destructive text-sm flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                {uploadStatus.error}
              </div>
            )}

            {uploadStatus?.status === "completed" && (
              <div className="text-green-600 text-sm flex items-center gap-1">
                <CheckCircle2 className="h-4 w-4" />
                Generated {uploadStatus.pairs_generated} Q&A pairs
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tips */}
      <Card className="bg-muted/50">
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground">
            <strong>Tip:</strong> PDF Q&A generation uses your local vLLM model. For best results with Romanian documents,
            use a multilingual model like Qwen or a Romanian-fine-tuned model.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default ImportEvalDatasetsPage;
