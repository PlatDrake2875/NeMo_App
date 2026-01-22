import PropTypes from "prop-types";
import {
  ArrowLeft,
  Database,
  Download,
  FolderOpen,
  GitBranch,
  LayoutDashboard,
} from "lucide-react";
import { Button } from "./ui/button";
import { DatasetDashboard } from "./rag-benchmark-hub/DatasetDashboard";
import { RawDatasetManager } from "./rag-benchmark-hub/raw-datasets/RawDatasetManager";
import { PreprocessingPipeline } from "./rag-benchmark-hub/preprocessing/PreprocessingPipeline";
import { ProcessedDatasetEditor } from "./rag-benchmark-hub/processed-datasets/ProcessedDatasetEditor";
import { GenerateQAPage } from "./rag-benchmark-hub/evaluation/GenerateQAPage";
import { EvaluationPage } from "./rag-benchmark-hub/evaluation/EvaluationPage";
import { EvaluationDashboard } from "./rag-benchmark-hub/evaluation/EvaluationDashboard";
import { AnnotationWorkbench } from "./rag-benchmark-hub/annotation/AnnotationWorkbench";
import { HuggingFaceImporter } from "./rag-benchmark-hub/huggingface/HuggingFaceImporter";

export function RAGBenchmarkHub({ onBack, isDarkMode, currentView, onViewChange }) {
  const renderView = () => {
    switch (currentView) {
      case "dashboard":
        return <DatasetDashboard onNavigate={onViewChange} />;
      case "raw":
        return <RawDatasetManager />;
      case "pipeline":
        return <PreprocessingPipeline />;
      case "processed":
        return <ProcessedDatasetEditor />;
      case "generate-qa":
        return <GenerateQAPage />;
      case "evaluation":
        return <EvaluationPage />;
      case "annotation":
        return <AnnotationWorkbench />;
      case "huggingface":
        return <HuggingFaceImporter />;
      default:
        return <DatasetDashboard onNavigate={onViewChange} />;
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-4 p-4 border-b bg-background">
        <Button
          variant="ghost"
          size="icon"
          onClick={onBack}
          className="md:hidden"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold">
          {currentView === "dashboard" && "RAG Benchmark Hub"}
          {currentView === "raw" && "Raw Datasets"}
          {currentView === "pipeline" && "Preprocessing Pipeline"}
          {currentView === "processed" && "Processed Datasets"}
          {currentView === "generate-qa" && "Generate Q&A Dataset"}
          {currentView === "evaluation" && "Evaluation"}
          {currentView === "annotation" && "Annotation Workbench"}
          {currentView === "huggingface" && "HuggingFace Import"}
        </h1>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-auto p-4 bg-muted/30">
        {renderView()}
      </div>
    </div>
  );
}

RAGBenchmarkHub.propTypes = {
  onBack: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool,
  currentView: PropTypes.string.isRequired,
  onViewChange: PropTypes.func.isRequired,
};

export default RAGBenchmarkHub;
