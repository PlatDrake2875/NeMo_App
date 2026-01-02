// frontend/src/components/guardrails/GuardrailsTester.jsx
import PropTypes from "prop-types";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Alert } from "../ui/alert";
import { Separator } from "../ui/separator";
import {
  Play,
  Trash2,
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  Loader2,
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useGuardrailsTesting } from "../../hooks/useGuardrailsTesting";

/**
 * GuardrailsTester - Test interface for guardrails configurations
 */
export function GuardrailsTester({ agentName, className }) {
  const {
    testInput,
    setTestInput,
    testResult,
    isTesting,
    testError,
    testHistory,
    runTest,
    clearResults,
    clearHistory,
  } = useGuardrailsTesting();

  const handleRunTest = async () => {
    if (!agentName || !testInput.trim()) return;
    await runTest(agentName, testInput);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      handleRunTest();
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Test Input Section */}
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-sm">Test Input</h3>
          {testResult && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearResults}
            >
              Clear
            </Button>
          )}
        </div>
        <div className="space-y-2">
          <textarea
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter a message to test the guardrails..."
            disabled={!agentName || isTesting}
            rows={3}
            className={cn(
              "w-full resize-none rounded-md border border-input bg-background px-3 py-2",
              "text-sm placeholder:text-muted-foreground",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              "disabled:cursor-not-allowed disabled:opacity-50"
            )}
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              Press Ctrl+Enter to run test
            </span>
            <Button
              onClick={handleRunTest}
              disabled={!agentName || !testInput.trim() || isTesting}
              size="sm"
            >
              {isTesting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  Testing...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-1" />
                  Run Test
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Test Result Section */}
      <div className="flex-1 overflow-y-auto">
        {testError && (
          <Alert variant="destructive" className="m-4">
            <AlertCircle className="h-4 w-4" />
            <span className="ml-2 text-sm">{testError}</span>
          </Alert>
        )}

        {testResult && (
          <div className="p-4 space-y-4">
            <div className="flex items-center gap-2">
              {testResult.blocked ? (
                <Badge variant="destructive" className="flex items-center gap-1">
                  <XCircle className="h-3 w-3" />
                  Blocked
                </Badge>
              ) : testResult.modified ? (
                <Badge variant="secondary" className="flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  Modified
                </Badge>
              ) : (
                <Badge variant="default" className="flex items-center gap-1 bg-green-600">
                  <CheckCircle className="h-3 w-3" />
                  Passed
                </Badge>
              )}
              {testResult.execution_time && (
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {testResult.execution_time}ms
                </span>
              )}
            </div>

            {/* Output */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Output</h4>
              <div className="bg-muted rounded-md p-3 text-sm">
                {testResult.output || testResult.response || "No output"}
              </div>
            </div>

            {/* Triggered Rails */}
            {testResult.triggered_rails && testResult.triggered_rails.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Triggered Rails</h4>
                <div className="flex flex-wrap gap-1">
                  {testResult.triggered_rails.map((rail, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {rail}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Details */}
            {testResult.details && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Details</h4>
                <pre className="bg-muted rounded-md p-3 text-xs overflow-x-auto">
                  {JSON.stringify(testResult.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Test History */}
        {testHistory.length > 0 && (
          <>
            <Separator />
            <div className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">Recent Tests</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearHistory}
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>
              <div className="space-y-2">
                {testHistory.slice(0, 5).map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-start gap-2 p-2 rounded-md bg-muted/50 text-xs"
                  >
                    {entry.status === "running" ? (
                      <Loader2 className="h-3 w-3 animate-spin text-primary mt-0.5" />
                    ) : entry.status === "error" ? (
                      <XCircle className="h-3 w-3 text-destructive mt-0.5" />
                    ) : entry.result?.blocked ? (
                      <XCircle className="h-3 w-3 text-destructive mt-0.5" />
                    ) : (
                      <CheckCircle className="h-3 w-3 text-green-600 mt-0.5" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="truncate">{entry.input}</p>
                      <p className="text-muted-foreground">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Empty State */}
        {!testResult && !testError && testHistory.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Play className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground">
              Enter a message and click "Run Test" to test the guardrails
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

GuardrailsTester.propTypes = {
  agentName: PropTypes.string,
  className: PropTypes.string,
};
