/**
 * useModelManager - Hook for managing vLLM model state and switching.
 *
 * Features:
 * - Fetch available models from vLLM
 * - Fetch cached models from HF cache
 * - Track model switch progress
 * - Disable chat during model switch
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { getApiBaseUrl } from "../lib/api-config";

export function useModelManager() {
  // Available models (currently loaded in vLLM)
  const [availableModels, setAvailableModels] = useState([]);
  // Cached models (in HF cache, can be switched to)
  const [cachedModels, setCachedModels] = useState([]);
  // Currently selected model
  const [selectedModel, setSelectedModel] = useState("");
  // Loading states
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState(null);
  // Current switch operation
  const [switchStatus, setSwitchStatus] = useState(null);

  // Polling interval ref
  const pollIntervalRef = useRef(null);

  /**
   * Fetch available models from vLLM
   */
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);

    try {
      const response = await fetch(`${getApiBaseUrl()}/api/models`);
      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }

      const data = await response.json();
      const modelNames = data.map((m) => m.name);
      setAvailableModels(modelNames);

      // Set selected model from localStorage or first available
      const stored = localStorage.getItem("selectedModel");
      if (stored && modelNames.includes(stored)) {
        setSelectedModel(stored);
      } else if (modelNames.length > 0) {
        setSelectedModel(modelNames[0]);
        localStorage.setItem("selectedModel", modelNames[0]);
      }
    } catch (err) {
      console.error("Error fetching models:", err);
      setModelsError(err.message);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  /**
   * Fetch cached models from HF cache
   */
  const fetchCachedModels = useCallback(async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/models/cached`);
      if (response.ok) {
        const data = await response.json();
        setCachedModels(data);
      }
    } catch (err) {
      console.error("Error fetching cached models:", err);
    }
  }, []);

  /**
   * Check for ongoing switch operation
   */
  const checkSwitchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/models/switch/status`);
      if (!response.ok) {
        setSwitchStatus(null);
        return;
      }

      const data = await response.json();
      setSwitchStatus(data);

      // Check if switch is in progress
      const inProgressStatuses = [
        "pending",
        "checking",
        "downloading",
        "stopping",
        "starting",
        "loading",
      ];

      if (data && inProgressStatuses.includes(data.status)) {
        // Start polling if not already
        if (!pollIntervalRef.current) {
          pollIntervalRef.current = setInterval(checkSwitchStatus, 2000);
        }
      } else {
        // Stop polling
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }

        // Refresh models if switch just completed
        if (data?.status === "ready") {
          fetchModels();
          fetchCachedModels();
          // Clear switch status after a delay
          setTimeout(() => setSwitchStatus(null), 3000);
        }
      }
    } catch (err) {
      console.error("Error checking switch status:", err);
      setSwitchStatus(null);
    }
  }, [fetchModels, fetchCachedModels]);

  /**
   * Start a model switch
   */
  const switchModel = useCallback(
    async (modelId) => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/api/models/switch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: modelId }),
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Failed to switch model");
        }

        const data = await response.json();
        setSwitchStatus(data);

        // Start polling for progress
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
        }
        pollIntervalRef.current = setInterval(checkSwitchStatus, 2000);

        return data;
      } catch (err) {
        console.error("Error switching model:", err);
        throw err;
      }
    },
    [checkSwitchStatus]
  );

  /**
   * Cancel an in-progress switch
   */
  const cancelSwitch = useCallback(async () => {
    if (!switchStatus?.id) return;

    try {
      const response = await fetch(
        `${getApiBaseUrl()}/api/models/switch/${switchStatus.id}`,
        { method: "DELETE" }
      );

      if (response.ok) {
        setSwitchStatus(null);
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      }
    } catch (err) {
      console.error("Error cancelling switch:", err);
    }
  }, [switchStatus]);

  /**
   * Update selected model (when user picks from dropdown without switching)
   */
  const handleModelSelect = useCallback((modelId) => {
    setSelectedModel(modelId);
    localStorage.setItem("selectedModel", modelId);
  }, []);

  // Initialize on mount
  useEffect(() => {
    fetchModels();
    fetchCachedModels();
    checkSwitchStatus();

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fetchModels, fetchCachedModels, checkSwitchStatus]);

  // Is chat disabled during certain switch phases?
  const isSwitching =
    switchStatus &&
    ["stopping", "starting", "loading"].includes(switchStatus.status);

  const isChatDisabled =
    switchStatus &&
    ["stopping", "starting", "loading"].includes(switchStatus.status);

  return {
    // State
    availableModels,
    cachedModels,
    selectedModel,
    modelsLoading,
    modelsError,
    switchStatus,
    isSwitching,
    isChatDisabled,

    // Actions
    setSelectedModel: handleModelSelect,
    fetchModels,
    fetchCachedModels,
    switchModel,
    cancelSwitch,
  };
}
