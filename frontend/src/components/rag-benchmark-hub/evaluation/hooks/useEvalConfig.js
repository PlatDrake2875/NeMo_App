import { useState, useCallback, useEffect } from "react";

/**
 * Hook for managing evaluation configuration state
 */
export function useEvalConfig(initialConfig = {}) {
  const [config, setConfig] = useState({
    collection: "",
    enableRag: true,
    embedder: "sentence-transformers/all-MiniLM-L6-v2",
    reranker: "colbert",
    enableReranking: true,
    topK: 5,
    temperature: 0.1,
    datasetId: null,
    ...initialConfig,
  });

  const [configHash, setConfigHash] = useState(null);
  const [isDirty, setIsDirty] = useState(false);

  // Update a single config value
  const updateConfig = useCallback((key, value) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
    setIsDirty(true);
    setConfigHash(null);
  }, []);

  // Bulk update config
  const setFullConfig = useCallback((newConfig) => {
    setConfig(newConfig);
    setIsDirty(true);
    setConfigHash(null);
  }, []);

  // Reset to initial config
  const resetConfig = useCallback(() => {
    setConfig({
      collection: "",
      enableRag: true,
      embedder: "sentence-transformers/all-MiniLM-L6-v2",
      reranker: "colbert",
      enableReranking: true,
      topK: 5,
      temperature: 0.1,
      datasetId: null,
      ...initialConfig,
    });
    setIsDirty(false);
    setConfigHash(null);
  }, [initialConfig]);

  // Convert to API format
  const toApiFormat = useCallback(() => {
    return {
      collection_name: config.collection,
      use_rag: config.enableRag,
      embedder: config.embedder,
      use_colbert: config.reranker === "colbert" && config.enableReranking,
      top_k: config.topK,
      temperature: config.temperature,
      eval_dataset_id: config.datasetId,
    };
  }, [config]);

  // Compute hash (would typically be done server-side)
  const computeHash = useCallback(async () => {
    const str = JSON.stringify({
      embedder_model: config.embedder,
      reranker_strategy: config.reranker === "colbert" && config.enableReranking ? "colbert" : "none",
      top_k: config.topK,
      temperature: config.temperature,
      dataset_id: config.datasetId,
      collection_name: config.collection,
      use_rag: config.enableRag,
    });

    // Simple hash for client-side preview
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    const hashStr = Math.abs(hash).toString(16).padStart(8, "0").slice(0, 12);
    setConfigHash(hashStr);
    return hashStr;
  }, [config]);

  return {
    config,
    updateConfig,
    setFullConfig,
    resetConfig,
    toApiFormat,
    configHash,
    computeHash,
    isDirty,
  };
}

export default useEvalConfig;
