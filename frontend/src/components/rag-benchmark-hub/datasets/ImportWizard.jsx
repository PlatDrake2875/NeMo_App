import { useState } from "react";
import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { Label } from "../../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import { Alert, AlertDescription } from "../../ui/alert";
import { Badge } from "../../ui/badge";
import {
  AlertCircle,
  CheckCircle,
  FileUp,
  Loader2,
  Upload,
} from "lucide-react";
import { API_BASE_URL } from "../../../lib/api-config";

const IMPORT_FORMATS = [
  {
    value: "squad",
    label: "SQuAD",
    description: "Stanford Question Answering Dataset (JSON)",
    extensions: [".json"],
  },
  {
    value: "natural_questions",
    label: "Natural Questions",
    description: "Google Natural Questions (JSON/JSONL)",
    extensions: [".json", ".jsonl"],
  },
  {
    value: "msmarco",
    label: "MS MARCO",
    description: "Microsoft MARCO (JSON/TSV)",
    extensions: [".json", ".jsonl", ".tsv"],
  },
];

/**
 * Wizard for importing evaluation datasets from standard formats
 */
export function ImportWizard({ onImportComplete }) {
  const [step, setStep] = useState(1);
  const [format, setFormat] = useState("");
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [maxPairs, setMaxPairs] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const selectedFormat = IMPORT_FORMATS.find((f) => f.value === format);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      if (!name) {
        setName(selectedFile.name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  const handleImport = async () => {
    if (!format || !name || !file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const params = new URLSearchParams({
        format,
        name,
        ...(maxPairs && { max_pairs: maxPairs }),
      });

      // Read file as bytes
      const fileBytes = await file.arrayBuffer();

      const response = await fetch(
        `${API_BASE_URL}/api/evaluation/datasets/import?${params}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
          },
          body: fileBytes,
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Import failed");
      }

      const data = await response.json();
      setResult(data);
      setStep(3);
      onImportComplete?.(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setStep(1);
    setFormat("");
    setName("");
    setFile(null);
    setMaxPairs("");
    setError(null);
    setResult(null);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Import Evaluation Dataset
        </CardTitle>
        <CardDescription>
          Import Q&A pairs from standard dataset formats
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Step indicators */}
        <div className="flex items-center justify-center gap-2 mb-6">
          {[1, 2, 3].map((s) => (
            <div
              key={s}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                s === step
                  ? "bg-primary text-primary-foreground"
                  : s < step
                  ? "bg-green-500 text-white"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {s < step ? <CheckCircle className="h-4 w-4" /> : s}
            </div>
          ))}
        </div>

        {/* Step 1: Select format */}
        {step === 1 && (
          <div className="space-y-4">
            <Label>Select Format</Label>
            <div className="grid gap-3">
              {IMPORT_FORMATS.map((f) => (
                <div
                  key={f.value}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    format === f.value
                      ? "border-primary bg-primary/5"
                      : "hover:border-muted-foreground/50"
                  }`}
                  onClick={() => setFormat(f.value)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{f.label}</p>
                      <p className="text-sm text-muted-foreground">
                        {f.description}
                      </p>
                    </div>
                    <div className="flex gap-1">
                      {f.extensions.map((ext) => (
                        <Badge key={ext} variant="outline" className="text-xs">
                          {ext}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <Button
              className="w-full"
              onClick={() => setStep(2)}
              disabled={!format}
            >
              Continue
            </Button>
          </div>
        )}

        {/* Step 2: Upload file */}
        {step === 2 && (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Dataset Name</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Evaluation Dataset"
              />
            </div>

            <div className="space-y-2">
              <Label>Select File</Label>
              <div
                className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                onClick={() => document.getElementById("file-input")?.click()}
              >
                <input
                  id="file-input"
                  type="file"
                  className="hidden"
                  accept={selectedFormat?.extensions.join(",")}
                  onChange={handleFileSelect}
                />
                <FileUp className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                {file ? (
                  <p className="text-sm font-medium">{file.name}</p>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Click to select a {selectedFormat?.label} file
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Max Pairs (optional)</Label>
              <Input
                type="number"
                value={maxPairs}
                onChange={(e) => setMaxPairs(e.target.value)}
                placeholder="Import all pairs"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to import all pairs
              </p>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setStep(1)}>
                Back
              </Button>
              <Button
                className="flex-1"
                onClick={handleImport}
                disabled={!name || !file || loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  "Import Dataset"
                )}
              </Button>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="space-y-4">
            <div className="text-center py-4">
              <CheckCircle className="h-12 w-12 mx-auto text-green-500 mb-3" />
              <h3 className="font-semibold text-lg">Import Successful!</h3>
              <p className="text-muted-foreground">
                {result.pair_count} Q&A pairs imported from {result.source_format}
              </p>
            </div>

            <div className="bg-muted rounded-lg p-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Dataset ID</span>
                <span className="font-mono">{result.id}</span>
              </div>
              <div className="flex justify-between">
                <span>Name</span>
                <span>{result.name}</span>
              </div>
              <div className="flex justify-between">
                <span>Pairs Imported</span>
                <span>{result.pair_count}</span>
              </div>
              {result.validation && (
                <>
                  <div className="flex justify-between">
                    <span>Validation</span>
                    <span className={result.validation.valid ? "text-green-600" : "text-yellow-600"}>
                      {result.validation.valid ? "Passed" : `${result.validation.warning_count} warnings`}
                    </span>
                  </div>
                </>
              )}
            </div>

            <Button className="w-full" onClick={handleReset}>
              Import Another Dataset
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

ImportWizard.propTypes = {
  onImportComplete: PropTypes.func,
};

export default ImportWizard;
