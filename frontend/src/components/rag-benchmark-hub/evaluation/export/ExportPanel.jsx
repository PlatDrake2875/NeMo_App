import { useState } from "react";
import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../../ui/card";
import { Button } from "../../../ui/button";
import { Label } from "../../../ui/label";
import { Checkbox } from "../../../ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../ui/select";
import { Download, FileJson, FileSpreadsheet, FileText } from "lucide-react";
import { getApiBaseUrl } from "../../../../lib/api-config";

const EXPORT_FORMATS = [
  {
    value: "csv",
    label: "CSV",
    description: "Comma-separated values, opens in Excel/Sheets",
    icon: FileSpreadsheet,
  },
  {
    value: "json",
    label: "JSON",
    description: "Full data with all details",
    icon: FileJson,
  },
  {
    value: "latex",
    label: "LaTeX Table",
    description: "Ready for academic papers",
    icon: FileText,
  },
];

/**
 * Export panel for evaluation results
 */
export function ExportPanel({ runId, runData }) {
  const [format, setFormat] = useState("csv");
  const [options, setOptions] = useState({
    includeDetails: true,
    includeChunks: false,
    includeSummary: true,
  });

  const handleExport = () => {
    if (format === "csv") {
      window.open(`${getApiBaseUrl()}/api/evaluation/runs/${runId}/csv`, "_blank");
    } else if (format === "json") {
      exportAsJSON();
    } else if (format === "latex") {
      exportAsLaTeX();
    }
  };

  const exportAsJSON = () => {
    const data = { ...runData };

    if (!options.includeChunks) {
      data.results = data.results?.map((r) => {
        const { retrieved_chunks, ...rest } = r;
        return rest;
      });
    }

    if (!options.includeDetails) {
      data.results = data.results?.map((r) => {
        const { score_details, ...rest } = r;
        return rest;
      });
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `evaluation_${runId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsLaTeX = () => {
    const metrics = runData?.metrics || {};

    let latex = `\\begin{table}[h]
\\centering
\\caption{RAG Evaluation Results - ${runData?.name || runId}}
\\begin{tabular}{lc}
\\hline
\\textbf{Metric} & \\textbf{Score} \\\\
\\hline
Answer Correctness & ${((metrics.answer_correctness || 0) * 100).toFixed(1)}\\% \\\\
Faithfulness & ${((metrics.faithfulness || 0) * 100).toFixed(1)}\\% \\\\
Context Precision & ${((metrics.context_precision || 0) * 100).toFixed(1)}\\% \\\\
Answer Relevancy & ${((metrics.answer_relevancy || 0) * 100).toFixed(1)}\\% \\\\
\\hline
Avg. Latency & ${(metrics.avg_latency || 0).toFixed(2)}s \\\\
\\hline
\\end{tabular}
\\label{tab:rag-eval}
\\end{table}`;

    if (options.includeSummary) {
      latex += `

% Configuration
% Collection: ${runData?.config?.collection || "N/A"}
% Embedder: ${runData?.config?.embedder || "N/A"}
% Reranker: ${runData?.config?.use_colbert ? "ColBERT" : "None"}
% Top-K: ${runData?.config?.top_k || 5}
% Total Pairs: ${runData?.results?.length || 0}`;
    }

    const blob = new Blob([latex], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `evaluation_${runId}.tex`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectedFormat = EXPORT_FORMATS.find((f) => f.value === format);
  const FormatIcon = selectedFormat?.icon || Download;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Download className="h-5 w-5" />
          Export Results
        </CardTitle>
        <CardDescription>
          Download evaluation results in your preferred format
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Format selection */}
        <div className="space-y-2">
          <Label>Export Format</Label>
          <div className="grid gap-2">
            {EXPORT_FORMATS.map((f) => {
              const Icon = f.icon;
              return (
                <div
                  key={f.value}
                  className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-colors ${
                    format === f.value
                      ? "border-primary bg-primary/5"
                      : "hover:border-muted-foreground/50"
                  }`}
                  onClick={() => setFormat(f.value)}
                >
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">{f.label}</p>
                    <p className="text-xs text-muted-foreground">
                      {f.description}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Options */}
        {format === "json" && (
          <div className="space-y-3">
            <Label>Options</Label>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeDetails"
                  checked={options.includeDetails}
                  onCheckedChange={(checked) =>
                    setOptions((prev) => ({ ...prev, includeDetails: checked }))
                  }
                />
                <label htmlFor="includeDetails" className="text-sm">
                  Include score details
                </label>
              </div>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="includeChunks"
                  checked={options.includeChunks}
                  onCheckedChange={(checked) =>
                    setOptions((prev) => ({ ...prev, includeChunks: checked }))
                  }
                />
                <label htmlFor="includeChunks" className="text-sm">
                  Include retrieved chunks
                </label>
              </div>
            </div>
          </div>
        )}

        {format === "latex" && (
          <div className="space-y-3">
            <Label>Options</Label>
            <div className="flex items-center gap-2">
              <Checkbox
                id="includeSummary"
                checked={options.includeSummary}
                onCheckedChange={(checked) =>
                  setOptions((prev) => ({ ...prev, includeSummary: checked }))
                }
              />
              <label htmlFor="includeSummary" className="text-sm">
                Include configuration comments
              </label>
            </div>
          </div>
        )}

        {/* Export button */}
        <Button className="w-full" onClick={handleExport}>
          <FormatIcon className="h-4 w-4 mr-2" />
          Export as {selectedFormat?.label}
        </Button>
      </CardContent>
    </Card>
  );
}

ExportPanel.propTypes = {
  runId: PropTypes.string.isRequired,
  runData: PropTypes.object,
};

export default ExportPanel;
