import { useCallback, useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Hook for managing documents - CRUD operations and state.
 * @param {string} dataset - Optional dataset filter
 * @returns {{
 *   documents: Array,
 *   sources: Array,
 *   stats: object | null,
 *   isLoading: boolean,
 *   error: string | null,
 *   uploadProgress: number,
 *   isUploading: boolean,
 *   fetchDocuments: () => Promise<void>,
 *   fetchSources: () => Promise<void>,
 *   fetchStats: () => Promise<void>,
 *   uploadDocument: (file: File, config: object) => Promise<object>,
 *   deleteDocument: (filename: string) => Promise<void>,
 *   refreshAll: () => Promise<void>
 * }}
 */
export function useDocumentManager(dataset = null) {
  const [documents, setDocuments] = useState([]);
  const [sources, setSources] = useState([]);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const url = dataset
        ? `${API_BASE_URL}/api/documents?dataset=${encodeURIComponent(dataset)}`
        : `${API_BASE_URL}/api/documents`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      if (data && Array.isArray(data.documents)) {
        // Sort by source then page
        const sortedDocs = data.documents.sort((a, b) => {
          const sourceA = a.metadata?.original_filename || a.metadata?.source || "";
          const sourceB = b.metadata?.original_filename || b.metadata?.source || "";
          if (sourceA < sourceB) return -1;
          if (sourceA > sourceB) return 1;
          const pageA = a.metadata?.page ?? -1;
          const pageB = b.metadata?.page ?? -1;
          return pageA - pageB;
        });
        setDocuments(sortedDocs);
      } else {
        setDocuments([]);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
      setError(err.message);
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  }, [dataset]);

  const fetchSources = useCallback(async () => {
    try {
      const url = dataset
        ? `${API_BASE_URL}/api/documents/sources?dataset=${encodeURIComponent(dataset)}`
        : `${API_BASE_URL}/api/documents/sources`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setSources(data.sources || []);
    } catch (err) {
      console.error("Error fetching sources:", err);
      // Don't set error - this is a secondary fetch
    }
  }, [dataset]);

  const fetchStats = useCallback(async () => {
    try {
      const url = dataset
        ? `${API_BASE_URL}/api/documents/stats?dataset=${encodeURIComponent(dataset)}`
        : `${API_BASE_URL}/api/documents/stats`;

      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error("Error fetching stats:", err);
      // Calculate stats from documents as fallback
      if (documents.length > 0) {
        const uniqueSources = new Set();
        documents.forEach(doc => {
          uniqueSources.add(doc.metadata?.original_filename || doc.metadata?.source || "Unknown");
        });
        setStats({
          total_documents: uniqueSources.size,
          total_chunks: documents.length
        });
      }
    }
  }, [dataset, documents]);

  const uploadDocument = useCallback(async (file, config = {}) => {
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Add chunking config
      if (config.chunkingMethod) {
        formData.append("chunking_method", config.chunkingMethod);
      }
      if (config.chunkSize) {
        formData.append("chunk_size", config.chunkSize.toString());
      }
      if (config.chunkOverlap) {
        formData.append("chunk_overlap", config.chunkOverlap.toString());
      }
      if (dataset) {
        formData.append("dataset", dataset);
      }

      // Use XMLHttpRequest for progress tracking
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(progress);
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            const response = JSON.parse(xhr.responseText);
            setIsUploading(false);
            setUploadProgress(100);
            resolve(response);
          } else {
            const error = new Error(`Upload failed: ${xhr.status}`);
            setError(error.message);
            setIsUploading(false);
            reject(error);
          }
        });

        xhr.addEventListener("error", () => {
          const error = new Error("Upload failed: Network error");
          setError(error.message);
          setIsUploading(false);
          reject(error);
        });

        xhr.open("POST", `${API_BASE_URL}/api/upload`);
        xhr.send(formData);
      });
    } catch (err) {
      console.error("Error uploading document:", err);
      setError(err.message);
      setIsUploading(false);
      throw err;
    }
  }, [dataset]);

  const deleteDocument = useCallback(async (filename) => {
    setError(null);
    try {
      const url = dataset
        ? `${API_BASE_URL}/api/documents/source/${encodeURIComponent(filename)}?dataset=${encodeURIComponent(dataset)}`
        : `${API_BASE_URL}/api/documents/source/${encodeURIComponent(filename)}`;

      const response = await fetch(url, { method: "DELETE" });

      if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.status}`);
      }

      // Refresh documents after deletion
      await fetchDocuments();
      await fetchSources();
      await fetchStats();
    } catch (err) {
      console.error("Error deleting document:", err);
      setError(err.message);
      throw err;
    }
  }, [dataset, fetchDocuments, fetchSources, fetchStats]);

  const refreshAll = useCallback(async () => {
    setIsLoading(true);
    try {
      await Promise.all([fetchDocuments(), fetchSources(), fetchStats()]);
    } finally {
      setIsLoading(false);
    }
  }, [fetchDocuments, fetchSources, fetchStats]);

  // Fetch documents and sources on mount
  useEffect(() => {
    fetchDocuments();
    fetchSources();
  }, [fetchDocuments, fetchSources]);

  // Calculate stats when documents change
  useEffect(() => {
    fetchStats();
  }, [documents, fetchStats]);

  return {
    documents,
    sources,
    stats,
    isLoading,
    error,
    uploadProgress,
    isUploading,
    fetchDocuments,
    fetchSources,
    fetchStats,
    uploadDocument,
    deleteDocument,
    refreshAll
  };
}
