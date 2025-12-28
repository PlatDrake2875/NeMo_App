import { useEffect, useState } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { ScrollArea } from "../../ui/scroll-area";
import { Switch } from "../../ui/switch";
import { Slider } from "../../ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Play,
} from "lucide-react";
import { cn } from "../../../lib/utils";
import { API_BASE_URL } from "../../../lib/api-config";

export function PreprocessingPipeline() {
  // Source dataset selection
  const [rawDatasets, setRawDatasets] = useState([]);
  const [selectedRawDatasetId, setSelectedRawDatasetId] = useState("");
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);

  // Output configuration
  const [outputName, setOutputName] = useState("");
  const [outputDescription, setOutputDescription] = useState("");

  // Cleaning configuration
  const [cleaningEnabled, setCleaningEnabled] = useState(false);
  const [removeHeaders, setRemoveHeaders] = useState(false);
  const [removePageNumbers, setRemovePageNumbers] = useState(false);
  const [normalizeWhitespace, setNormalizeWhitespace] = useState(true);
  // New cleaning options
  const [removeHtmlMarkup, setRemoveHtmlMarkup] = useState(false);
  const [removeUrls, setRemoveUrls] = useState(false);
  const [removeCitations, setRemoveCitations] = useState(false);
  const [removeEmails, setRemoveEmails] = useState(false);
  const [removePhoneNumbers, setRemovePhoneNumbers] = useState(false);
  const [normalizeUnicode, setNormalizeUnicode] = useState(false);
  const [preserveCodeBlocks, setPreserveCodeBlocks] = useState(true);

  // Lightweight Metadata configuration (no LLM)
  const [lightweightMetadataEnabled, setLightweightMetadataEnabled] = useState(false);
  const [extractRakeKeywords, setExtractRakeKeywords] = useState(false);
  const [extractStatistics, setExtractStatistics] = useState(false);
  const [detectLanguage, setDetectLanguage] = useState(false);
  const [extractSpacyEntities, setExtractSpacyEntities] = useState(false);

  // LLM Metadata configuration
  const [metadataEnabled, setMetadataEnabled] = useState(false);
  const [extractSummary, setExtractSummary] = useState(true);
  const [extractKeywords, setExtractKeywords] = useState(true);
  const [extractEntities, setExtractEntities] = useState(true);
  const [extractCategories, setExtractCategories] = useState(true);

  // Chunking configuration
  const [chunkingMethod, setChunkingMethod] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);

  // Vector store configuration
  const [vectorBackend, setVectorBackend] = useState("pgvector");
  const [embedderModel, setEmbedderModel] = useState("all-MiniLM-L6-v2");

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRawDatasets();
  }, []);

  const fetchRawDatasets = async () => {
    setIsLoadingDatasets(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/raw-datasets`);
      if (!response.ok) throw new Error("Failed to fetch datasets");
      const data = await response.json();
      setRawDatasets(data.datasets || []);
    } catch (err) {
      console.error("Error fetching datasets:", err);
      setError(err.message);
    } finally {
      setIsLoadingDatasets(false);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedRawDatasetId || !outputName.trim()) {
      setError("Please select a source dataset and provide an output name");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProcessingStatus({ step: "Creating dataset...", progress: 0 });

    try {
      // Create the processed dataset
      const response = await fetch(
        `${API_BASE_URL}/api/processed-datasets?start_processing=true`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: outputName.trim(),
            description: outputDescription.trim() || null,
            raw_dataset_id: parseInt(selectedRawDatasetId),
            embedder_config: {
              model_name: embedderModel,
              model_type: "huggingface",
            },
            preprocessing_config: {
              cleaning: {
                enabled: cleaningEnabled,
                remove_headers_footers: removeHeaders,
                remove_page_numbers: removePageNumbers,
                normalize_whitespace: normalizeWhitespace,
                remove_html_markup: removeHtmlMarkup,
                remove_urls: removeUrls,
                remove_citations: removeCitations,
                remove_emails: removeEmails,
                remove_phone_numbers: removePhoneNumbers,
                normalize_unicode: normalizeUnicode,
                preserve_code_blocks: preserveCodeBlocks,
              },
              lightweight_metadata: {
                enabled: lightweightMetadataEnabled,
                extract_rake_keywords: extractRakeKeywords,
                extract_statistics: extractStatistics,
                detect_language: detectLanguage,
                extract_spacy_entities: extractSpacyEntities,
              },
              llm_metadata: {
                enabled: metadataEnabled,
                extract_summary: extractSummary,
                extract_keywords: extractKeywords,
                extract_entities: extractEntities,
                extract_categories: extractCategories,
              },
              chunking: {
                method: chunkingMethod,
                chunk_size: chunkSize,
                chunk_overlap: chunkOverlap,
              },
            },
            vector_backend: vectorBackend,
          }),
        }
      );

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to create dataset");
      }

      const result = await response.json();
      setProcessingStatus({
        step: "Processing started",
        progress: 100,
        datasetId: result.id,
      });

      // Reset form
      setOutputName("");
      setOutputDescription("");
      setSelectedRawDatasetId("");
    } catch (err) {
      console.error("Processing error:", err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const selectedDataset = rawDatasets.find(
    (ds) => ds.id.toString() === selectedRawDatasetId
  );

  return (
    <ScrollArea className="h-[calc(100vh-200px)]">
      <div className="space-y-3 pr-4">
        {/* Source Dataset */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">Source Dataset</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <Select
              value={selectedRawDatasetId}
              onValueChange={setSelectedRawDatasetId}
              disabled={isLoadingDatasets}
            >
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Select a raw dataset..." />
              </SelectTrigger>
              <SelectContent>
                {rawDatasets.map((ds) => (
                  <SelectItem key={ds.id} value={ds.id.toString()}>
                    {ds.name} ({ds.total_file_count} files)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedDataset && (
              <p className="text-xs text-muted-foreground mt-1">
                {selectedDataset.total_file_count} files,{" "}
                {(selectedDataset.total_size_bytes / 1024 / 1024).toFixed(1)} MB
              </p>
            )}
          </CardContent>
        </Card>

        {/* Cleaning (Optional) */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 py-3">
            <CardTitle className="text-base">1. Cleaning</CardTitle>
            <Switch checked={cleaningEnabled} onCheckedChange={setCleaningEnabled} />
          </CardHeader>
          {cleaningEnabled && (
            <CardContent className="pt-0 space-y-3">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Document Structure</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removeHeaders} onCheckedChange={setRemoveHeaders} className="scale-90" />
                  Headers/footers
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removePageNumbers} onCheckedChange={setRemovePageNumbers} className="scale-90" />
                  Page numbers
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={normalizeWhitespace} onCheckedChange={setNormalizeWhitespace} className="scale-90" />
                  Normalize whitespace
                </label>
              </div>

              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">Content Removal</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removeHtmlMarkup} onCheckedChange={setRemoveHtmlMarkup} className="scale-90" />
                  HTML/Markup
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removeUrls} onCheckedChange={setRemoveUrls} className="scale-90" />
                  URLs
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removeCitations} onCheckedChange={setRemoveCitations} className="scale-90" />
                  Citations
                </label>
              </div>

              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">Privacy</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removeEmails} onCheckedChange={setRemoveEmails} className="scale-90" />
                  Emails
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={removePhoneNumbers} onCheckedChange={setRemovePhoneNumbers} className="scale-90" />
                  Phone numbers
                </label>
              </div>

              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">Formatting</p>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={normalizeUnicode} onCheckedChange={setNormalizeUnicode} className="scale-90" />
                  Normalize Unicode
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={preserveCodeBlocks} onCheckedChange={setPreserveCodeBlocks} className="scale-90" />
                  Preserve code blocks
                </label>
              </div>
            </CardContent>
          )}
        </Card>

        {/* Lightweight Metadata (Optional) */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 py-3">
            <div>
              <CardTitle className="text-base">2. Quick Metadata</CardTitle>
              <CardDescription className="text-xs">No LLM required</CardDescription>
            </div>
            <Switch checked={lightweightMetadataEnabled} onCheckedChange={setLightweightMetadataEnabled} />
          </CardHeader>
          {lightweightMetadataEnabled && (
            <CardContent className="pt-0">
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractRakeKeywords} onCheckedChange={setExtractRakeKeywords} className="scale-90" />
                  RAKE Keywords
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractStatistics} onCheckedChange={setExtractStatistics} className="scale-90" />
                  Statistics
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={detectLanguage} onCheckedChange={setDetectLanguage} className="scale-90" />
                  Language
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractSpacyEntities} onCheckedChange={setExtractSpacyEntities} className="scale-90" />
                  Entities (spaCy)
                </label>
              </div>
            </CardContent>
          )}
        </Card>

        {/* LLM Metadata (Optional) */}
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 py-3">
            <div>
              <CardTitle className="text-base">3. LLM Metadata</CardTitle>
              <CardDescription className="text-xs">Slower, more accurate</CardDescription>
            </div>
            <Switch checked={metadataEnabled} onCheckedChange={setMetadataEnabled} />
          </CardHeader>
          {metadataEnabled && (
            <CardContent className="pt-0">
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractSummary} onCheckedChange={setExtractSummary} className="scale-90" />
                  Summaries
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractKeywords} onCheckedChange={setExtractKeywords} className="scale-90" />
                  Keywords
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractEntities} onCheckedChange={setExtractEntities} className="scale-90" />
                  Entities
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <Switch checked={extractCategories} onCheckedChange={setExtractCategories} className="scale-90" />
                  Categories
                </label>
              </div>
            </CardContent>
          )}
        </Card>

        {/* Chunking (Required) */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">4. Chunking</CardTitle>
          </CardHeader>
          <CardContent className="pt-0 space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">Method</Label>
              <Select value={chunkingMethod} onValueChange={setChunkingMethod}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="recursive">Recursive (natural boundaries)</SelectItem>
                  <SelectItem value="fixed">Fixed Size</SelectItem>
                  <SelectItem value="semantic">Semantic</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Chunk Size</Label>
                  <Input
                    type="number"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(Math.max(100, Math.min(8000, parseInt(e.target.value) || 100)))}
                    className="w-20 h-7 text-xs text-right"
                    min={100}
                    max={8000}
                  />
                </div>
                <Slider
                  value={[chunkSize]}
                  onValueChange={([v]) => setChunkSize(v)}
                  min={100}
                  max={8000}
                  step={50}
                  className="py-1"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Overlap</Label>
                  <Input
                    type="number"
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(Math.max(0, Math.min(2000, parseInt(e.target.value) || 0)))}
                    className="w-20 h-7 text-xs text-right"
                    min={0}
                    max={2000}
                  />
                </div>
                <Slider
                  value={[chunkOverlap]}
                  onValueChange={([v]) => setChunkOverlap(v)}
                  min={0}
                  max={2000}
                  step={10}
                  className="py-1"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Vector Database (Required) */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">5. Vector Database</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-xs">Backend</Label>
                <Select value={vectorBackend} onValueChange={setVectorBackend}>
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pgvector">pgvector</SelectItem>
                    <SelectItem value="qdrant">Qdrant</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label className="text-xs">Embedding Model</Label>
                <Select value={embedderModel} onValueChange={setEmbedderModel}>
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all-MiniLM-L6-v2">MiniLM-L6 (384d)</SelectItem>
                    <SelectItem value="BAAI/bge-small-en-v1.5">BGE-small (384d)</SelectItem>
                    <SelectItem value="nomic-ai/nomic-embed-text-v1">Nomic (768d)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Output Configuration */}
        <Card>
          <CardHeader className="py-3">
            <CardTitle className="text-base">Output</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-xs">Dataset Name *</Label>
                <Input
                  placeholder="my-dataset"
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  className="h-9"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Description</Label>
                <Input
                  placeholder="Optional..."
                  value={outputDescription}
                  onChange={(e) => setOutputDescription(e.target.value)}
                  className="h-9"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 text-destructive p-3 bg-destructive/10 rounded text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Success Display */}
        {processingStatus?.progress === 100 && (
          <div className="flex items-center gap-2 text-green-600 p-3 bg-green-100 dark:bg-green-900/20 rounded text-sm">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            Processing started! Check the Processed Datasets tab.
          </div>
        )}

        {/* Start Processing Button */}
        <Button
          className="w-full"
          onClick={handleStartProcessing}
          disabled={isProcessing || !selectedRawDatasetId || !outputName.trim()}
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Start Processing
            </>
          )}
        </Button>
      </div>
    </ScrollArea>
  );
}

export default PreprocessingPipeline;
