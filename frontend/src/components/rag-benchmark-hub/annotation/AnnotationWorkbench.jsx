import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import { Progress } from "../../ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import {
  Check,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Edit2,
  Keyboard,
  List,
  SkipForward,
  X,
  XCircle,
} from "lucide-react";
import { getApiBaseUrl } from "../../../lib/api-config";
import { QAPairEditor } from "./QAPairEditor";
import { AnnotationQueue } from "./AnnotationQueue";
import { AnnotationStats } from "./AnnotationStats";

/**
 * Human-in-the-loop annotation workbench for reviewing and editing Q&A pairs
 */
export function AnnotationWorkbench({ datasetId }) {
  const [dataset, setDataset] = useState(null);
  const [pairs, setPairs] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [annotations, setAnnotations] = useState({}); // { index: { status, difficulty, notes } }
  const [showQueue, setShowQueue] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);

  // Fetch dataset
  const fetchDataset = useCallback(async () => {
    if (!datasetId) return;

    setLoading(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/evaluation/datasets/${datasetId}`);
      if (!response.ok) throw new Error("Failed to fetch dataset");
      const data = await response.json();
      setDataset(data);
      setPairs(data.pairs || []);

      // Initialize annotations
      const initialAnnotations = {};
      (data.pairs || []).forEach((_, idx) => {
        initialAnnotations[idx] = { status: "pending", difficulty: "medium", notes: "" };
      });
      setAnnotations(initialAnnotations);
    } catch (err) {
      console.error("Error fetching dataset:", err);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    fetchDataset();
  }, [fetchDataset]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      switch (e.key) {
        case "a":
        case "1":
          handleApprove();
          break;
        case "r":
        case "2":
          handleReject();
          break;
        case "s":
        case "3":
          handleSkip();
          break;
        case "ArrowLeft":
        case "k":
          handlePrevious();
          break;
        case "ArrowRight":
        case "j":
          handleNext();
          break;
        case "e":
          // Focus edit mode (would need ref)
          break;
        case "?":
          setShowKeyboardHelp((prev) => !prev);
          break;
        default:
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentIndex, pairs.length]);

  const handleApprove = () => {
    setAnnotations((prev) => ({
      ...prev,
      [currentIndex]: { ...prev[currentIndex], status: "approved" },
    }));
    if (currentIndex < pairs.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  const handleReject = () => {
    setAnnotations((prev) => ({
      ...prev,
      [currentIndex]: { ...prev[currentIndex], status: "rejected" },
    }));
    if (currentIndex < pairs.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  const handleSkip = () => {
    if (currentIndex < pairs.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  };

  const handleNext = () => {
    if (currentIndex < pairs.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  const handleEditPair = (updatedPair) => {
    setPairs((prev) =>
      prev.map((p, i) => (i === currentIndex ? { ...p, ...updatedPair } : p))
    );
  };

  const handleSetDifficulty = (difficulty) => {
    setAnnotations((prev) => ({
      ...prev,
      [currentIndex]: { ...prev[currentIndex], difficulty },
    }));
  };

  const handleSaveAnnotations = async () => {
    // Save approved pairs back to dataset
    const approvedPairs = pairs.filter((_, idx) => annotations[idx]?.status === "approved");

    // In a real implementation, this would update the dataset
    console.log("Saving annotations:", approvedPairs.length, "approved pairs");
    alert(`${approvedPairs.length} pairs approved and ready to save`);
  };

  // Calculate stats
  const stats = {
    total: pairs.length,
    approved: Object.values(annotations).filter((a) => a.status === "approved").length,
    rejected: Object.values(annotations).filter((a) => a.status === "rejected").length,
    pending: Object.values(annotations).filter((a) => a.status === "pending").length,
  };

  const progress = pairs.length > 0 ? ((stats.approved + stats.rejected) / pairs.length) * 100 : 0;
  const currentPair = pairs[currentIndex];
  const currentAnnotation = annotations[currentIndex] || { status: "pending", difficulty: "medium" };

  if (loading) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-muted-foreground">
          Loading dataset...
        </CardContent>
      </Card>
    );
  }

  if (!dataset || pairs.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-muted-foreground">
          {datasetId ? "No pairs found in dataset" : "Select a dataset to annotate"}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Edit2 className="h-5 w-5" />
            Annotation Workbench
          </h2>
          <p className="text-muted-foreground text-sm">
            Review and annotate Q&A pairs for {dataset.name}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowQueue(!showQueue)}
          >
            <List className="h-4 w-4 mr-2" />
            Queue
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowKeyboardHelp(!showKeyboardHelp)}
          >
            <Keyboard className="h-4 w-4 mr-2" />
            Shortcuts
          </Button>
          <Button onClick={handleSaveAnnotations}>
            Save Annotations
          </Button>
        </div>
      </div>

      {/* Progress */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Progress: {stats.approved + stats.rejected} / {stats.total}</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <Progress value={progress} />
            <div className="flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1 text-green-600">
                <CheckCircle className="h-3 w-3" />
                {stats.approved} approved
              </span>
              <span className="flex items-center gap-1 text-red-600">
                <XCircle className="h-3 w-3" />
                {stats.rejected} rejected
              </span>
              <span className="text-muted-foreground">
                {stats.pending} pending
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Keyboard shortcuts help */}
      {showKeyboardHelp && (
        <Card className="border-primary/30 bg-primary/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Keyboard Shortcuts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">A</kbd> / <kbd className="px-1.5 py-0.5 bg-muted rounded">1</kbd> - Approve</div>
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">R</kbd> / <kbd className="px-1.5 py-0.5 bg-muted rounded">2</kbd> - Reject</div>
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">S</kbd> / <kbd className="px-1.5 py-0.5 bg-muted rounded">3</kbd> - Skip</div>
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">←</kbd> / <kbd className="px-1.5 py-0.5 bg-muted rounded">K</kbd> - Previous</div>
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">→</kbd> / <kbd className="px-1.5 py-0.5 bg-muted rounded">J</kbd> - Next</div>
              <div><kbd className="px-1.5 py-0.5 bg-muted rounded">?</kbd> - Toggle help</div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main annotation area */}
        <div className="lg:col-span-2 space-y-4">
          {/* Navigation */}
          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevious}
              disabled={currentIndex === 0}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              {currentIndex + 1} of {pairs.length}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleNext}
              disabled={currentIndex === pairs.length - 1}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>

          {/* Q&A Editor */}
          <QAPairEditor
            pair={currentPair}
            annotation={currentAnnotation}
            onEdit={handleEditPair}
            onSetDifficulty={handleSetDifficulty}
          />

          {/* Action buttons */}
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="lg"
              className="border-red-500/50 text-red-600 hover:bg-red-500/10"
              onClick={handleReject}
            >
              <X className="h-5 w-5 mr-2" />
              Reject (R)
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={handleSkip}
            >
              <SkipForward className="h-5 w-5 mr-2" />
              Skip (S)
            </Button>
            <Button
              size="lg"
              className="bg-green-600 hover:bg-green-700"
              onClick={handleApprove}
            >
              <Check className="h-5 w-5 mr-2" />
              Approve (A)
            </Button>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {showQueue ? (
            <AnnotationQueue
              pairs={pairs}
              annotations={annotations}
              currentIndex={currentIndex}
              onSelectIndex={setCurrentIndex}
            />
          ) : (
            <AnnotationStats stats={stats} />
          )}
        </div>
      </div>
    </div>
  );
}

AnnotationWorkbench.propTypes = {
  datasetId: PropTypes.string,
};

export default AnnotationWorkbench;
