import { useCallback, useEffect, useState } from "react";
import { getApiBaseUrl } from "../lib/api-config";

/**
 * Hook for managing vector store configuration and status.
 * @returns {{
 *   backend: string,
 *   connectionStatus: 'connected' | 'disconnected' | 'checking' | 'unknown',
 *   config: object | null,
 *   embeddingModel: string,
 *   availableEmbeddingModels: string[],
 *   isLoading: boolean,
 *   error: string | null,
 *   fetchConfig: () => Promise<void>,
 *   checkHealth: () => Promise<void>
 * }}
 */
export function useVectorStoreConfig() {
  const [backend, setBackend] = useState("pgvector");
  const [connectionStatus, setConnectionStatus] = useState("unknown");
  const [config, setConfig] = useState(null);
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Fetch vector store config
      const configResponse = await fetch(`${getApiBaseUrl()}/api/config/vector-store`);
      if (configResponse.ok) {
        const configData = await configResponse.json();
        setConfig(configData);
        setBackend(configData.backend || "pgvector");
        setEmbeddingModel(configData.embedding_model || "");
      }

      // Try to fetch available embedding models
      try {
        const modelsResponse = await fetch(`${getApiBaseUrl()}/api/config/embedding-models`);
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          setAvailableEmbeddingModels(modelsData.models || []);
        }
      } catch {
        // Fallback to common embedding models
        setAvailableEmbeddingModels([
          "sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/all-mpnet-base-v2",
          "BAAI/bge-small-en-v1.5",
          "BAAI/bge-base-en-v1.5",
          "nomic-ai/nomic-embed-text-v1"
        ]);
      }
    } catch (err) {
      console.error("Error fetching vector store config:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const checkHealth = useCallback(async () => {
    setConnectionStatus("checking");
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/health`);
      if (response.ok) {
        const data = await response.json();
        // Check if vector store is healthy
        if (data.pg_status === "ok" || data.vector_store_status === "ok") {
          setConnectionStatus("connected");
        } else {
          setConnectionStatus("disconnected");
        }
      } else {
        setConnectionStatus("disconnected");
      }
    } catch (err) {
      console.error("Error checking health:", err);
      setConnectionStatus("disconnected");
    }
  }, []);

  // Fetch config and check health on mount
  useEffect(() => {
    fetchConfig();
    checkHealth();
  }, [fetchConfig, checkHealth]);

  return {
    backend,
    connectionStatus,
    config,
    embeddingModel,
    availableEmbeddingModels,
    isLoading,
    error,
    fetchConfig,
    checkHealth
  };
}
