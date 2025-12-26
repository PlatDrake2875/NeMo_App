import { useState } from "react";
import PropTypes from "prop-types";
import { ArrowLeft } from "lucide-react";
import { Button } from "./ui/button";
import { RAGHubSidebar } from "./rag-benchmark-hub/RAGHubSidebar";
import { DatasetDashboard } from "./rag-benchmark-hub/DatasetDashboard";
import { RawDatasetManager } from "./rag-benchmark-hub/raw-datasets/RawDatasetManager";
import { PreprocessingPipeline } from "./rag-benchmark-hub/preprocessing/PreprocessingPipeline";
import { ProcessedDatasetEditor } from "./rag-benchmark-hub/processed-datasets/ProcessedDatasetEditor";
import { HuggingFaceImporter } from "./rag-benchmark-hub/huggingface/HuggingFaceImporter";

export function RAGBenchmarkHub({ onBack, isDarkMode, toggleTheme }) {
  const [currentView, setCurrentView] = useState("dashboard");

  const renderView = () => {
    switch (currentView) {
      case "dashboard":
        return <DatasetDashboard onNavigate={setCurrentView} />;
      case "raw":
        return <RawDatasetManager isDarkMode={isDarkMode} />;
      case "pipeline":
        return <PreprocessingPipeline />;
      case "processed":
        return <ProcessedDatasetEditor />;
      case "huggingface":
        return <HuggingFaceImporter />;
      default:
        return <DatasetDashboard onNavigate={setCurrentView} />;
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* Hub Sidebar */}
      <RAGHubSidebar
        currentView={currentView}
        onViewChange={setCurrentView}
        onBack={onBack}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />

      {/* Main Content */}
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
            {currentView === "huggingface" && "HuggingFace Import"}
          </h1>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto p-4 bg-muted/30">
          {renderView()}
        </div>
      </div>
    </div>
  );
}

RAGBenchmarkHub.propTypes = {
  onBack: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool,
  toggleTheme: PropTypes.func,
};

export default RAGBenchmarkHub;
