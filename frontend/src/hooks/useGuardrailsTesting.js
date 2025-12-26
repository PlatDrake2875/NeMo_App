// frontend/src/hooks/useGuardrailsTesting.js
import { useState, useCallback } from "react";
import { API_BASE_URL } from "../lib/api-config";

/**
 * Hook for testing Guardrails configurations
 */
export function useGuardrailsTesting() {
  const [testInput, setTestInput] = useState("");
  const [testResult, setTestResult] = useState(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testError, setTestError] = useState(null);
  const [testHistory, setTestHistory] = useState([]);

  /**
   * Run a test against an agent
   */
  const runTest = useCallback(async (agentName, input) => {
    if (!agentName || !input.trim()) return;

    setIsTesting(true);
    setTestError(null);
    setTestResult(null);

    const testEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      input: input.trim(),
      agentName,
      status: "running",
    };

    // Add to history
    setTestHistory((prev) => [testEntry, ...prev].slice(0, 20)); // Keep last 20

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${agentName}/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: input.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Test failed: ${response.status}`);
      }

      const result = await response.json();
      setTestResult(result);

      // Update history entry
      setTestHistory((prev) =>
        prev.map((entry) =>
          entry.id === testEntry.id
            ? { ...entry, status: "completed", result }
            : entry
        )
      );

      return result;
    } catch (err) {
      console.error("Error running test:", err);
      setTestError(err.message);

      // Update history entry with error
      setTestHistory((prev) =>
        prev.map((entry) =>
          entry.id === testEntry.id
            ? { ...entry, status: "error", error: err.message }
            : entry
        )
      );

      return null;
    } finally {
      setIsTesting(false);
    }
  }, []);

  /**
   * Clear test results
   */
  const clearResults = useCallback(() => {
    setTestResult(null);
    setTestError(null);
  }, []);

  /**
   * Clear test history
   */
  const clearHistory = useCallback(() => {
    setTestHistory([]);
  }, []);

  return {
    testInput,
    setTestInput,
    testResult,
    isTesting,
    testError,
    testHistory,
    runTest,
    clearResults,
    clearHistory,
  };
}
