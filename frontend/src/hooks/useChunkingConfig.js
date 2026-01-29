import { useCallback, useEffect, useState } from "react";
import { getApiBaseUrl } from "../lib/api-config";

/**
 * Hook for managing chunking configuration state.
 * @returns {{
 *   method: string,
 *   setMethod: (method: string) => void,
 *   chunkSize: number,
 *   setChunkSize: (size: number) => void,
 *   chunkOverlap: number,
 *   setChunkOverlap: (overlap: number) => void,
 *   availableMethods: object,
 *   isLoading: boolean,
 *   error: string | null,
 *   fetchAvailableMethods: () => Promise<void>,
 *   resetToDefaults: () => void
 * }}
 */
export function useChunkingConfig() {
  const [method, setMethod] = useState("recursive");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [availableMethods, setAvailableMethods] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAvailableMethods = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/chunking/methods`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setAvailableMethods(data.methods || data);
    } catch (err) {
      console.error("Error fetching chunking methods:", err);
      setError(err.message);
      // Fallback to default methods
      setAvailableMethods({
        recursive: {
          name: "Recursive",
          description: "Recursively splits text at natural boundaries"
        },
        fixed: {
          name: "Fixed Size",
          description: "Splits text into fixed-size chunks"
        },
        semantic: {
          name: "Semantic",
          description: "Splits based on semantic meaning"
        }
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const resetToDefaults = useCallback(() => {
    setMethod("recursive");
    setChunkSize(1000);
    setChunkOverlap(200);
  }, []);

  // Fetch available methods on mount
  useEffect(() => {
    fetchAvailableMethods();
  }, [fetchAvailableMethods]);

  return {
    method,
    setMethod,
    chunkSize,
    setChunkSize,
    chunkOverlap,
    setChunkOverlap,
    availableMethods,
    isLoading,
    error,
    fetchAvailableMethods,
    resetToDefaults
  };
}
