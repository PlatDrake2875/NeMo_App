import { useEffect, useState, useMemo } from "react";
import { Button } from "../../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../ui/card";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import { ScrollArea } from "../../ui/scroll-area";
import { Switch } from "../../ui/switch";
import { Badge } from "../../ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../ui/tooltip";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Play,
  HelpCircle,
  Zap,
  Settings,
  Sparkles,
  Save,
  Download,
  Upload,
} from "lucide-react";
import { cn } from "../../../lib/utils";
import { getApiBaseUrl } from "../../../lib/api-config";
import { useTemplates } from "../../../hooks/useTemplates";

// Configuration presets - chunking/embedding now happens at evaluation time
const PRESETS = {
  quick: {
    name: "Quick",
    description: "Minimal processing, fastest",
    icon: Zap,
    config: {
      cleaning: { enabled: false },
      lightweightMetadata: { enabled: false },
      llmMetadata: { enabled: false },
    },
  },
  standard: {
    name: "Standard",
    description: "Balanced quality and speed",
    icon: Settings,
    config: {
      cleaning: {
        enabled: true,
        removeHeaders: false,
        removePageNumbers: true,
        normalizeWhitespace: true,
        removeHtmlMarkup: true,
        removeUrls: false,
        removeCitations: false,
        removeEmails: false,
        removePhoneNumbers: false,
        normalizeUnicode: true,
        preserveCodeBlocks: true,
      },
      lightweightMetadata: {
        enabled: true,
        extractRakeKeywords: true,
        extractStatistics: true,
        detectLanguage: true,
        extractSpacyEntities: false,
      },
      llmMetadata: { enabled: false },
    },
  },
  thorough: {
    name: "Thorough",
    description: "Maximum quality, slower",
    icon: Sparkles,
    config: {
      cleaning: {
        enabled: true,
        removeHeaders: true,
        removePageNumbers: true,
        normalizeWhitespace: true,
        removeHtmlMarkup: true,
        removeUrls: false,
        removeCitations: false,
        removeEmails: true,
        removePhoneNumbers: true,
        normalizeUnicode: true,
        preserveCodeBlocks: true,
      },
      lightweightMetadata: {
        enabled: true,
        extractRakeKeywords: true,
        extractStatistics: true,
        detectLanguage: true,
        extractSpacyEntities: true,
      },
      llmMetadata: {
        enabled: true,
        extractSummary: true,
        extractKeywords: true,
        extractEntities: true,
        extractCategories: true,
        model: "meta-llama/Llama-3.2-3B-Instruct",
      },
    },
  },
};

// LLM models for metadata extraction
const LLM_MODELS = [
  { value: "meta-llama/Llama-3.2-3B-Instruct", label: "Llama 3.2 3B (Fast)" },
  { value: "meta-llama/Llama-3.1-8B-Instruct", label: "Llama 3.1 8B (Better)" },
];

// Help texts for cleaning options
const HELP_TEXTS = {
  removeHeaders: "Removes short lines (<30 chars) typically appearing as headers/footers in PDFs",
  removePageNumbers: "Removes standalone numbers on their own lines (e.g., page numbers)",
  normalizeWhitespace: "Collapses multiple spaces/newlines into single spaces. Runs AFTER header/page number removal",
  removeHtmlMarkup: "Strips HTML tags while preserving text content",
  removeUrls: "Removes http/https URLs from text",
  removeCitations: "Removes academic citation patterns like [1], (Smith 2020)",
  removeEmails: "Removes email addresses from text",
  removePhoneNumbers: "Removes phone number patterns",
  normalizeUnicode: "Normalizes Unicode characters (e.g., fancy quotes to standard quotes)",
  preserveCodeBlocks: "Protects code blocks (``` fenced) from cleaning modifications",
  extractRakeKeywords: "Extracts keywords using RAKE algorithm (fast, rule-based)",
  extractStatistics: "Calculates word count, sentence count, avg word length, etc.",
  detectLanguage: "Detects the document's language",
  extractSpacyEntities: "Extracts named entities using spaCy NER (slower but accurate)",
};

// Step indicator configuration - chunking/embedding now at evaluation time
const STEPS = [
  { id: "source", label: "Source", required: true },
  { id: "cleaning", label: "Cleaning", required: false },
  { id: "metadata", label: "Metadata", required: false },
  { id: "output", label: "Output", required: true },
];

// Helper component for tooltips
function HelpTooltip({ text }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help ml-1" />
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        <p className="text-xs">{text}</p>
      </TooltipContent>
    </Tooltip>
  );
}

// Step Progress Indicator component
function StepIndicator({ currentStep, completedSteps }) {
  return (
    <div className="flex items-center justify-between mb-4 px-2">
      {STEPS.map((step, index) => {
        const isCompleted = completedSteps.includes(step.id);
        const isCurrent = currentStep === step.id;

        return (
          <div key={step.id} className="flex items-center">
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  "w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium border-2 transition-colors",
                  isCompleted
                    ? "bg-green-500 border-green-500 text-white"
                    : isCurrent
                    ? "bg-primary border-primary text-primary-foreground"
                    : "bg-background border-muted-foreground/30 text-muted-foreground"
                )}
              >
                {isCompleted ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  index + 1
                )}
              </div>
              <span
                className={cn(
                  "text-[10px] mt-1",
                  isCompleted || isCurrent
                    ? "text-foreground"
                    : "text-muted-foreground"
                )}
              >
                {step.label}
              </span>
              {!step.required && (
                <span className="text-[8px] text-muted-foreground">(optional)</span>
              )}
            </div>
            {index < STEPS.length - 1 && (
              <div
                className={cn(
                  "h-0.5 w-8 mx-1",
                  isCompleted ? "bg-green-500" : "bg-muted-foreground/30"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function PreprocessingPipeline({ onNavigate }) {
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
  const [llmModel, setLlmModel] = useState("meta-llama/Llama-3.2-3B-Instruct");

  // Note: Chunking and embedding configuration moved to evaluation page

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [error, setError] = useState(null);
  const [showSuccessDialog, setShowSuccessDialog] = useState(false);
  const [createdDatasetName, setCreatedDatasetName] = useState("");

  // Selected preset
  const [selectedPreset, setSelectedPreset] = useState(null);

  // Template management
  const {
    saveTemplate,
    exportTemplate,
    importTemplateFromFile,
    isLoading: isTemplateLoading,
    error: templateError,
  } = useTemplates();
  const [showSaveTemplateDialog, setShowSaveTemplateDialog] = useState(false);
  const [templateName, setTemplateName] = useState("");
  const [templateDescription, setTemplateDescription] = useState("");
  const [templateTags, setTemplateTags] = useState("");
  const fileInputRef = useState(null);

  // Calculate completed steps
  const completedSteps = useMemo(() => {
    const completed = [];
    if (selectedRawDatasetId) completed.push("source");
    if (cleaningEnabled) completed.push("cleaning");
    if (lightweightMetadataEnabled || metadataEnabled) completed.push("metadata");
    if (outputName.trim()) completed.push("output");
    return completed;
  }, [
    selectedRawDatasetId,
    cleaningEnabled,
    lightweightMetadataEnabled,
    metadataEnabled,
    outputName,
  ]);

  // Determine current step
  const currentStep = useMemo(() => {
    if (!selectedRawDatasetId) return "source";
    if (!outputName.trim()) return "output";
    return "output";
  }, [selectedRawDatasetId, outputName]);

  useEffect(() => {
    fetchRawDatasets();
  }, []);

  const fetchRawDatasets = async () => {
    setIsLoadingDatasets(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/raw-datasets`);
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

  const applyPreset = (presetKey) => {
    const preset = PRESETS[presetKey];
    if (!preset) return;

    setSelectedPreset(presetKey);
    const { config } = preset;

    // Apply cleaning config
    setCleaningEnabled(config.cleaning.enabled);
    if (config.cleaning.enabled) {
      setRemoveHeaders(config.cleaning.removeHeaders ?? false);
      setRemovePageNumbers(config.cleaning.removePageNumbers ?? false);
      setNormalizeWhitespace(config.cleaning.normalizeWhitespace ?? true);
      setRemoveHtmlMarkup(config.cleaning.removeHtmlMarkup ?? false);
      setRemoveUrls(config.cleaning.removeUrls ?? false);
      setRemoveCitations(config.cleaning.removeCitations ?? false);
      setRemoveEmails(config.cleaning.removeEmails ?? false);
      setRemovePhoneNumbers(config.cleaning.removePhoneNumbers ?? false);
      setNormalizeUnicode(config.cleaning.normalizeUnicode ?? false);
      setPreserveCodeBlocks(config.cleaning.preserveCodeBlocks ?? true);
    }

    // Apply lightweight metadata config
    setLightweightMetadataEnabled(config.lightweightMetadata.enabled);
    if (config.lightweightMetadata.enabled) {
      setExtractRakeKeywords(config.lightweightMetadata.extractRakeKeywords ?? false);
      setExtractStatistics(config.lightweightMetadata.extractStatistics ?? false);
      setDetectLanguage(config.lightweightMetadata.detectLanguage ?? false);
      setExtractSpacyEntities(config.lightweightMetadata.extractSpacyEntities ?? false);
    }

    // Apply LLM metadata config
    setMetadataEnabled(config.llmMetadata.enabled);
    if (config.llmMetadata.enabled) {
      setExtractSummary(config.llmMetadata.extractSummary ?? true);
      setExtractKeywords(config.llmMetadata.extractKeywords ?? true);
      setExtractEntities(config.llmMetadata.extractEntities ?? true);
      setExtractCategories(config.llmMetadata.extractCategories ?? true);
      setLlmModel(config.llmMetadata.model ?? "meta-llama/Llama-3.2-3B-Instruct");
    }
    // Note: Chunking config moved to evaluation page
  };

  const resetForm = () => {
    setOutputName("");
    setOutputDescription("");
    setSelectedRawDatasetId("");
    setProcessingStatus(null);
    setError(null);
    setShowSuccessDialog(false);
    setSelectedPreset(null);
  };

  // Get current configuration as a template object
  // Note: Chunking config is no longer included - it's configured at evaluation time
  const getCurrentConfigAsTemplate = () => ({
    name: templateName || "My Template",
    description: templateDescription || null,
    version: "1.0",
    tags: templateTags ? templateTags.split(",").map((t) => t.trim()).filter(Boolean) : [],
    preprocessing: {
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
        model: llmModel,
        extract_summary: extractSummary,
        extract_keywords: extractKeywords,
        extract_entities: extractEntities,
        extract_categories: extractCategories,
      },
    },
  });

  const handleSaveTemplate = async () => {
    if (!templateName.trim()) {
      setError("Please provide a template name");
      return;
    }
    try {
      const template = getCurrentConfigAsTemplate();
      await saveTemplate(template);
      setShowSaveTemplateDialog(false);
      setTemplateName("");
      setTemplateDescription("");
      setTemplateTags("");
    } catch (err) {
      setError(err.message);
    }
  };

  const handleExportCurrentConfig = () => {
    const template = getCurrentConfigAsTemplate();
    template.name = "exported_config";
    // Note: Chunking config is no longer included - it's configured at evaluation time
    const yamlContent = `# Exported Preprocessing Configuration
# Generated: ${new Date().toISOString()}
# Note: Chunking and embedding are now configured at evaluation time

name: ${template.name}
description: ${template.description || "Exported configuration"}
version: "${template.version}"
tags: ${JSON.stringify(template.tags)}

preprocessing:
  cleaning:
    enabled: ${template.preprocessing.cleaning.enabled}
    remove_headers_footers: ${template.preprocessing.cleaning.remove_headers_footers}
    remove_page_numbers: ${template.preprocessing.cleaning.remove_page_numbers}
    normalize_whitespace: ${template.preprocessing.cleaning.normalize_whitespace}
    remove_html_markup: ${template.preprocessing.cleaning.remove_html_markup}
    remove_urls: ${template.preprocessing.cleaning.remove_urls}
    remove_citations: ${template.preprocessing.cleaning.remove_citations}
    remove_emails: ${template.preprocessing.cleaning.remove_emails}
    remove_phone_numbers: ${template.preprocessing.cleaning.remove_phone_numbers}
    normalize_unicode: ${template.preprocessing.cleaning.normalize_unicode}
    preserve_code_blocks: ${template.preprocessing.cleaning.preserve_code_blocks}
  lightweight_metadata:
    enabled: ${template.preprocessing.lightweight_metadata.enabled}
    extract_rake_keywords: ${template.preprocessing.lightweight_metadata.extract_rake_keywords}
    extract_statistics: ${template.preprocessing.lightweight_metadata.extract_statistics}
    detect_language: ${template.preprocessing.lightweight_metadata.detect_language}
    extract_spacy_entities: ${template.preprocessing.lightweight_metadata.extract_spacy_entities}
  llm_metadata:
    enabled: ${template.preprocessing.llm_metadata.enabled}
    model: ${template.preprocessing.llm_metadata.model}
    extract_summary: ${template.preprocessing.llm_metadata.extract_summary}
    extract_keywords: ${template.preprocessing.llm_metadata.extract_keywords}
    extract_entities: ${template.preprocessing.llm_metadata.extract_entities}
    extract_categories: ${template.preprocessing.llm_metadata.extract_categories}
`;

    const blob = new Blob([yamlContent], { type: "application/x-yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "preprocessing_config.yaml";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImportConfig = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const result = await importTemplateFromFile(file);
        // If the imported template has preprocessing config, apply it
        if (result.preprocessing) {
          const config = {
            cleaning: result.preprocessing.cleaning || {},
            lightweightMetadata: result.preprocessing.lightweight_metadata || {},
            llmMetadata: result.preprocessing.llm_metadata || {},
            chunking: result.preprocessing.chunking || {},
          };

          // Apply cleaning
          if (config.cleaning) {
            setCleaningEnabled(config.cleaning.enabled ?? false);
            setRemoveHeaders(config.cleaning.remove_headers_footers ?? false);
            setRemovePageNumbers(config.cleaning.remove_page_numbers ?? false);
            setNormalizeWhitespace(config.cleaning.normalize_whitespace ?? true);
            setRemoveHtmlMarkup(config.cleaning.remove_html_markup ?? false);
            setRemoveUrls(config.cleaning.remove_urls ?? false);
            setRemoveCitations(config.cleaning.remove_citations ?? false);
            setRemoveEmails(config.cleaning.remove_emails ?? false);
            setRemovePhoneNumbers(config.cleaning.remove_phone_numbers ?? false);
            setNormalizeUnicode(config.cleaning.normalize_unicode ?? false);
            setPreserveCodeBlocks(config.cleaning.preserve_code_blocks ?? true);
          }

          // Apply lightweight metadata
          if (config.lightweightMetadata) {
            setLightweightMetadataEnabled(config.lightweightMetadata.enabled ?? false);
            setExtractRakeKeywords(config.lightweightMetadata.extract_rake_keywords ?? false);
            setExtractStatistics(config.lightweightMetadata.extract_statistics ?? false);
            setDetectLanguage(config.lightweightMetadata.detect_language ?? false);
            setExtractSpacyEntities(config.lightweightMetadata.extract_spacy_entities ?? false);
          }

          // Apply LLM metadata
          if (config.llmMetadata) {
            setMetadataEnabled(config.llmMetadata.enabled ?? false);
            setLlmModel(config.llmMetadata.model ?? "meta-llama/Llama-3.2-3B-Instruct");
            setExtractSummary(config.llmMetadata.extract_summary ?? true);
            setExtractKeywords(config.llmMetadata.extract_keywords ?? true);
            setExtractEntities(config.llmMetadata.extract_entities ?? true);
            setExtractCategories(config.llmMetadata.extract_categories ?? true);
          }
          // Note: Chunking config from imported templates is ignored - it's configured at evaluation time
        }
      } catch (err) {
        setError(`Failed to import config: ${err.message}`);
      }
    };
    reader.readAsText(file);
    // Reset the input
    event.target.value = "";
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
      // Note: chunking and embedding are now configured at evaluation time
      const response = await fetch(
        `${getApiBaseUrl()}/api/processed-datasets?start_processing=true`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: outputName.trim(),
            description: outputDescription.trim() || null,
            raw_dataset_id: parseInt(selectedRawDatasetId),
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
                model: llmModel,
                extract_summary: extractSummary,
                extract_keywords: extractKeywords,
                extract_entities: extractEntities,
                extract_categories: extractCategories,
              },
              // chunking: null - deferred to evaluation time
            },
          }),
        }
      );

      if (!response.ok) {
        const errData = await response.json();
        // Handle Pydantic validation errors (array) or simple error messages
        let errorMessage = "Failed to create dataset";
        if (errData.detail) {
          if (Array.isArray(errData.detail)) {
            // Pydantic validation errors
            errorMessage = errData.detail
              .map((e) => e.msg || e.message || JSON.stringify(e))
              .join("; ");
          } else if (typeof errData.detail === "string") {
            errorMessage = errData.detail;
          } else {
            errorMessage = JSON.stringify(errData.detail);
          }
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setProcessingStatus({
        step: "Processing started",
        progress: 100,
        datasetId: result.id,
      });
      setCreatedDatasetName(outputName.trim());
      setShowSuccessDialog(true);
    } catch (err) {
      console.error("Processing error:", err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleNavigateToProcessed = () => {
    setShowSuccessDialog(false);
    resetForm();
    if (onNavigate) {
      onNavigate("processed");
    }
  };

  const selectedDataset = rawDatasets.find(
    (ds) => ds.id.toString() === selectedRawDatasetId
  );

  return (
    <TooltipProvider>
      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="space-y-3 pr-4">
          {/* Step Progress Indicator */}
          <StepIndicator currentStep={currentStep} completedSteps={completedSteps} />

          {/* Configuration Presets */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-base">Quick Start Presets</CardTitle>
              <CardDescription className="text-xs">
                Choose a preset to quickly configure all settings
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(PRESETS).map(([key, preset]) => {
                  const Icon = preset.icon;
                  return (
                    <Button
                      key={key}
                      variant={selectedPreset === key ? "default" : "outline"}
                      className="h-auto py-2 flex flex-col items-center gap-1"
                      onClick={() => applyPreset(key)}
                    >
                      <Icon className="h-4 w-4" />
                      <span className="text-xs font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">
                        {preset.description}
                      </span>
                    </Button>
                  );
                })}
              </div>
              {/* Template actions */}
              <div className="flex gap-2 mt-3 pt-3 border-t">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setShowSaveTemplateDialog(true)}
                >
                  <Save className="h-3 w-3 mr-1" />
                  Save
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={handleExportCurrentConfig}
                >
                  <Download className="h-3 w-3 mr-1" />
                  Export
                </Button>
                <label className="flex-1">
                  <Button variant="outline" size="sm" className="w-full" asChild>
                    <span>
                      <Upload className="h-3 w-3 mr-1" />
                      Import
                    </span>
                  </Button>
                  <input
                    type="file"
                    accept=".yaml,.yml"
                    onChange={handleImportConfig}
                    className="hidden"
                  />
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Source Dataset */}
          <Card>
            <CardHeader className="py-3 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Source Dataset</CardTitle>
              <Badge variant="default">Required</Badge>
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
              <div className="flex items-center gap-2">
                <CardTitle className="text-base">1. Cleaning</CardTitle>
                <Badge variant="outline">Optional</Badge>
              </div>
              <Switch checked={cleaningEnabled} onCheckedChange={setCleaningEnabled} />
            </CardHeader>
            {cleaningEnabled && (
              <CardContent className="pt-0 space-y-3">
                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  Document Structure
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removeHeaders}
                      onCheckedChange={setRemoveHeaders}
                      className="scale-90"
                    />
                    Headers/footers
                    <HelpTooltip text={HELP_TEXTS.removeHeaders} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removePageNumbers}
                      onCheckedChange={setRemovePageNumbers}
                      className="scale-90"
                    />
                    Page numbers
                    <HelpTooltip text={HELP_TEXTS.removePageNumbers} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={normalizeWhitespace}
                      onCheckedChange={setNormalizeWhitespace}
                      className="scale-90"
                    />
                    Normalize whitespace
                    <HelpTooltip text={HELP_TEXTS.normalizeWhitespace} />
                  </label>
                </div>

                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">
                  Content Removal
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removeHtmlMarkup}
                      onCheckedChange={setRemoveHtmlMarkup}
                      className="scale-90"
                    />
                    HTML/Markup
                    <HelpTooltip text={HELP_TEXTS.removeHtmlMarkup} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removeUrls}
                      onCheckedChange={setRemoveUrls}
                      className="scale-90"
                    />
                    URLs
                    <HelpTooltip text={HELP_TEXTS.removeUrls} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removeCitations}
                      onCheckedChange={setRemoveCitations}
                      className="scale-90"
                    />
                    Citations
                    <HelpTooltip text={HELP_TEXTS.removeCitations} />
                  </label>
                </div>

                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">
                  Privacy
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removeEmails}
                      onCheckedChange={setRemoveEmails}
                      className="scale-90"
                    />
                    Emails
                    <HelpTooltip text={HELP_TEXTS.removeEmails} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={removePhoneNumbers}
                      onCheckedChange={setRemovePhoneNumbers}
                      className="scale-90"
                    />
                    Phone numbers
                    <HelpTooltip text={HELP_TEXTS.removePhoneNumbers} />
                  </label>
                </div>

                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider pt-1">
                  Formatting
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={normalizeUnicode}
                      onCheckedChange={setNormalizeUnicode}
                      className="scale-90"
                    />
                    Normalize Unicode
                    <HelpTooltip text={HELP_TEXTS.normalizeUnicode} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={preserveCodeBlocks}
                      onCheckedChange={setPreserveCodeBlocks}
                      className="scale-90"
                    />
                    Preserve code blocks
                    <HelpTooltip text={HELP_TEXTS.preserveCodeBlocks} />
                  </label>
                </div>
              </CardContent>
            )}
          </Card>

          {/* Lightweight Metadata (Optional) */}
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0 py-3">
              <div className="flex items-center gap-2">
                <div>
                  <CardTitle className="text-base">2. Quick Metadata</CardTitle>
                  <CardDescription className="text-xs">No LLM required</CardDescription>
                </div>
                <Badge variant="outline">Optional</Badge>
              </div>
              <Switch
                checked={lightweightMetadataEnabled}
                onCheckedChange={setLightweightMetadataEnabled}
              />
            </CardHeader>
            {lightweightMetadataEnabled && (
              <CardContent className="pt-0">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractRakeKeywords}
                      onCheckedChange={setExtractRakeKeywords}
                      className="scale-90"
                    />
                    RAKE Keywords
                    <HelpTooltip text={HELP_TEXTS.extractRakeKeywords} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractStatistics}
                      onCheckedChange={setExtractStatistics}
                      className="scale-90"
                    />
                    Statistics
                    <HelpTooltip text={HELP_TEXTS.extractStatistics} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={detectLanguage}
                      onCheckedChange={setDetectLanguage}
                      className="scale-90"
                    />
                    Language
                    <HelpTooltip text={HELP_TEXTS.detectLanguage} />
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractSpacyEntities}
                      onCheckedChange={setExtractSpacyEntities}
                      className="scale-90"
                    />
                    Entities (spaCy)
                    <HelpTooltip text={HELP_TEXTS.extractSpacyEntities} />
                  </label>
                </div>
              </CardContent>
            )}
          </Card>

          {/* LLM Metadata (Optional) */}
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0 py-3">
              <div className="flex items-center gap-2">
                <div>
                  <CardTitle className="text-base">3. LLM Metadata</CardTitle>
                  <CardDescription className="text-xs">Slower, more accurate</CardDescription>
                </div>
                <Badge variant="outline">Optional</Badge>
              </div>
              <Switch checked={metadataEnabled} onCheckedChange={setMetadataEnabled} />
            </CardHeader>
            {metadataEnabled && (
              <CardContent className="pt-0 space-y-3">
                <div className="space-y-1">
                  <Label className="text-xs">Model</Label>
                  <Select value={llmModel} onValueChange={setLlmModel}>
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {LLM_MODELS.map((model) => (
                        <SelectItem key={model.value} value={model.value}>
                          {model.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractSummary}
                      onCheckedChange={setExtractSummary}
                      className="scale-90"
                    />
                    Summaries
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractKeywords}
                      onCheckedChange={setExtractKeywords}
                      className="scale-90"
                    />
                    Keywords
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractEntities}
                      onCheckedChange={setExtractEntities}
                      className="scale-90"
                    />
                    Entities
                  </label>
                  <label className="flex items-center gap-2 text-sm cursor-pointer">
                    <Switch
                      checked={extractCategories}
                      onCheckedChange={setExtractCategories}
                      className="scale-90"
                    />
                    Categories
                  </label>
                </div>
              </CardContent>
            )}
          </Card>

          {/* Info: Chunking & Embedding at Evaluation Time */}
          <Card className="bg-muted/50 border-dashed">
            <CardContent className="pt-4 pb-4">
              <p className="text-sm text-muted-foreground">
                <strong>Note:</strong> Chunking and embedding configuration has been moved to the
                <strong> Evaluation</strong> page. This allows you to run multiple experiments with
                different configurations on the same preprocessed dataset.
              </p>
            </CardContent>
          </Card>

          {/* Output Configuration */}
          <Card>
            <CardHeader className="py-3 flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Output</CardTitle>
              <Badge variant="default">Required</Badge>
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

          {/* Start Processing Button */}
          <Button
            className="w-full"
            onClick={handleStartProcessing}
            disabled={
              isProcessing ||
              !selectedRawDatasetId ||
              !outputName.trim()
            }
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

      {/* Success Dialog */}
      <Dialog open={showSuccessDialog} onOpenChange={setShowSuccessDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-500" />
              Processing Started!
            </DialogTitle>
            <DialogDescription>
              Dataset &quot;{createdDatasetName}&quot; is being processed. You can monitor
              its progress in the Processed Datasets section.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => {
                setShowSuccessDialog(false);
                resetForm();
              }}
            >
              Process Another
            </Button>
            <Button onClick={handleNavigateToProcessed}>View Datasets</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Save Template Dialog */}
      <Dialog open={showSaveTemplateDialog} onOpenChange={setShowSaveTemplateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Save className="h-5 w-5" />
              Save Configuration as Template
            </DialogTitle>
            <DialogDescription>
              Save the current configuration for reuse. Templates can be loaded later or shared with others.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="template-name">Template Name *</Label>
              <Input
                id="template-name"
                placeholder="My Custom Template"
                value={templateName}
                onChange={(e) => setTemplateName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="template-description">Description</Label>
              <Input
                id="template-description"
                placeholder="Optional description..."
                value={templateDescription}
                onChange={(e) => setTemplateDescription(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="template-tags">Tags (comma-separated)</Label>
              <Input
                id="template-tags"
                placeholder="production, balanced, custom"
                value={templateTags}
                onChange={(e) => setTemplateTags(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowSaveTemplateDialog(false);
                setTemplateName("");
                setTemplateDescription("");
                setTemplateTags("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleSaveTemplate}
              disabled={!templateName.trim() || isTemplateLoading}
            >
              {isTemplateLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Template
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
}

export default PreprocessingPipeline;
