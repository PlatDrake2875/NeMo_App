import { useState, useCallback, useEffect } from "react";
import { API_BASE_URL } from "../../../../lib/api-config";

/**
 * Hook for fetching and managing evaluation runs
 */
export function useEvalRuns() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);
  const [selectedRunDetails, setSelectedRunDetails] = useState(null);

  // Fetch all runs
  const fetchRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs`);
      if (!response.ok) throw new Error("Failed to fetch runs");
      const data = await response.json();
      setRuns(data);
    } catch (err) {
      setError(err.message);
      console.error("Error fetching runs:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch a single run's details
  const fetchRunDetails = useCallback(async (runId) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`);
      if (!response.ok) throw new Error("Failed to fetch run details");
      const data = await response.json();
      setSelectedRun(runId);
      setSelectedRunDetails(data);
      return data;
    } catch (err) {
      setError(err.message);
      console.error("Error fetching run details:", err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Delete a run
  const deleteRun = useCallback(async (runId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("Failed to delete run");
      setRuns((prev) => prev.filter((r) => r.id !== runId));
      if (selectedRun === runId) {
        setSelectedRun(null);
        setSelectedRunDetails(null);
      }
      return true;
    } catch (err) {
      setError(err.message);
      console.error("Error deleting run:", err);
      return false;
    }
  }, [selectedRun]);

  // Re-score a run
  const rescoreRun = useCallback(async (runId, metrics = null) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/evaluation/runs/${runId}/rescore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metrics }),
      });
      if (!response.ok) throw new Error("Failed to re-score run");
      const data = await response.json();
      // Refresh runs and selected run
      await fetchRuns();
      if (selectedRun === runId) {
        await fetchRunDetails(runId);
      }
      return data;
    } catch (err) {
      setError(err.message);
      console.error("Error re-scoring run:", err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchRuns, fetchRunDetails, selectedRun]);

  // Export run as CSV
  const exportRunCSV = useCallback((runId) => {
    window.open(`${API_BASE_URL}/api/evaluation/runs/${runId}/csv`, "_blank");
  }, []);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedRun(null);
    setSelectedRunDetails(null);
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  return {
    runs,
    loading,
    error,
    selectedRun,
    selectedRunDetails,
    fetchRuns,
    fetchRunDetails,
    deleteRun,
    rescoreRun,
    exportRunCSV,
    clearSelection,
    selectRun: fetchRunDetails,
  };
}

export default useEvalRuns;
