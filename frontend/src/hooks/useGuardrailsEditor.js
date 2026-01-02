// frontend/src/hooks/useGuardrailsEditor.js
import { useState, useCallback, useEffect } from "react";
import { API_BASE_URL } from "../lib/api-config";

/**
 * Hook for managing Guardrails Editor state
 * Handles agent CRUD, config loading/saving
 */
export function useGuardrailsEditor() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [saveError, setSaveError] = useState(null);

  // Config file contents
  const [configYaml, setConfigYaml] = useState("");
  const [configColang, setConfigColang] = useState("");
  const [originalConfigYaml, setOriginalConfigYaml] = useState("");
  const [originalConfigColang, setOriginalConfigColang] = useState("");

  // Validation state
  const [validationResult, setValidationResult] = useState(null);
  const [isValidating, setIsValidating] = useState(false);

  /**
   * Fetch list of available agents
   */
  const fetchAgents = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agents: ${response.status}`);
      }
      const data = await response.json();
      setAgents(data.agents || []);
    } catch (err) {
      console.error("Error fetching agents:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Load agent configuration files
   */
  const loadAgentConfig = useCallback(async (agentName) => {
    if (!agentName) return;

    setIsLoading(true);
    setError(null);
    setValidationResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${agentName}/config`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to load config: ${response.status}`);
      }
      const data = await response.json();

      setConfigYaml(data.config_yaml || "");
      setConfigColang(data.config_colang || "");
      setOriginalConfigYaml(data.config_yaml || "");
      setOriginalConfigColang(data.config_colang || "");
      setSelectedAgent({
        name: agentName,
        metadata: data.metadata || {},
        is_custom: data.is_custom || false,
      });
    } catch (err) {
      console.error("Error loading agent config:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Save agent configuration files
   */
  const saveAgentConfig = useCallback(async () => {
    if (!selectedAgent) return;

    setIsSaving(true);
    setSaveError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${selectedAgent.name}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config_yaml: configYaml,
          config_colang: configColang,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to save config: ${response.status}`);
      }

      // Update original values after successful save
      setOriginalConfigYaml(configYaml);
      setOriginalConfigColang(configColang);

      return true;
    } catch (err) {
      console.error("Error saving agent config:", err);
      setSaveError(err.message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [selectedAgent, configYaml, configColang]);

  /**
   * Create a new agent
   */
  const createAgent = useCallback(async (name, description = "", baseAgent = null) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          base_agent: baseAgent,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to create agent: ${response.status}`);
      }

      await fetchAgents();
      await loadAgentConfig(name);
      return true;
    } catch (err) {
      console.error("Error creating agent:", err);
      setError(err.message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [fetchAgents, loadAgentConfig]);

  /**
   * Clone an existing agent
   */
  const cloneAgent = useCallback(async (sourceAgent, newName) => {
    return createAgent(newName, `Clone of ${sourceAgent}`, sourceAgent);
  }, [createAgent]);

  /**
   * Delete an agent
   */
  const deleteAgent = useCallback(async (agentName) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${agentName}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to delete agent: ${response.status}`);
      }

      // Clear selection if deleted agent was selected
      if (selectedAgent?.name === agentName) {
        setSelectedAgent(null);
        setConfigYaml("");
        setConfigColang("");
      }

      await fetchAgents();
      return true;
    } catch (err) {
      console.error("Error deleting agent:", err);
      setError(err.message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAgent, fetchAgents]);

  /**
   * Validate configuration content
   */
  const validateConfig = useCallback(async () => {
    if (!selectedAgent) return;

    setIsValidating(true);
    setValidationResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agents/${selectedAgent.name}/validate-content`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config_yaml: configYaml,
          config_colang: configColang,
        }),
      });

      const data = await response.json();
      setValidationResult(data);
      return data;
    } catch (err) {
      console.error("Error validating config:", err);
      setValidationResult({ valid: false, errors: [err.message] });
      return { valid: false, errors: [err.message] };
    } finally {
      setIsValidating(false);
    }
  }, [selectedAgent, configYaml, configColang]);

  /**
   * Check if there are unsaved changes
   */
  const hasUnsavedChanges = configYaml !== originalConfigYaml || configColang !== originalConfigColang;

  /**
   * Reset changes to original values
   */
  const resetChanges = useCallback(() => {
    setConfigYaml(originalConfigYaml);
    setConfigColang(originalConfigColang);
    setValidationResult(null);
    setSaveError(null);
  }, [originalConfigYaml, originalConfigColang]);

  // Load agents on mount
  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return {
    // State
    agents,
    selectedAgent,
    isLoading,
    isSaving,
    error,
    saveError,

    // Config content
    configYaml,
    setConfigYaml,
    configColang,
    setConfigColang,

    // Validation
    validationResult,
    isValidating,
    validateConfig,

    // Actions
    fetchAgents,
    loadAgentConfig,
    saveAgentConfig,
    createAgent,
    cloneAgent,
    deleteAgent,
    hasUnsavedChanges,
    resetChanges,
  };
}
