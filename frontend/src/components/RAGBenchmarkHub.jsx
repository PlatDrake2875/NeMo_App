import { useState } from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { DatasetDashboard } from "./rag-benchmark-hub/DatasetDashboard";
import { RawDatasetManager } from "./rag-benchmark-hub/raw-datasets/RawDatasetManager";
import { PreprocessingPipeline } from "./rag-benchmark-hub/preprocessing/PreprocessingPipeline";
import { ProcessedDatasetEditor } from "./rag-benchmark-hub/processed-datasets/ProcessedDatasetEditor";
import { HuggingFaceImporter } from "./rag-benchmark-hub/huggingface/HuggingFaceImporter";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "raw", label: "Raw Datasets", icon: FolderOpen },
  { id: "pipeline", label: "Preprocessing", icon: GitBranch },
  { id: "processed", label: "Processed Datasets", icon: Database },
  { id: "huggingface", label: "HuggingFace Import", icon: Download },
];

export function RAGBenchmarkHub({ onBack, isDarkMode }) {
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

  const currentNavItem = navItems.find((item) => item.id === currentView);
  const CurrentIcon = currentNavItem?.icon || LayoutDashboard;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header with navigation dropdown */}
      <div className="flex items-center gap-3 p-3 border-b bg-background">
        <Button
          variant="ghost"
          size="icon"
          onClick={onBack}
          title="Back to Chat"
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>

        <Select value={currentView} onValueChange={setCurrentView}>
          <SelectTrigger className="w-[220px]">
            <div className="flex items-center gap-2">
              <CurrentIcon className="h-4 w-4 shrink-0" />
              <SelectValue placeholder="Select view" />
            </div>
          </SelectTrigger>
          <SelectContent>
            {navItems.map((item) => (
              <SelectItem key={item.id} value={item.id}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <span className="text-sm text-muted-foreground hidden sm:inline">
          RAG Benchmark Hub
        </span>
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
};

export default RAGBenchmarkHub;
