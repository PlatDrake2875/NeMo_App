import { useState, useCallback } from "react";
import { API_BASE_URL } from "../lib/api-config";

/**
 * Custom hook for managing experiment templates.
 * Provides functionality to load, save, import, and export templates.
 */
export function useTemplates() {
  const [templates, setTemplates] = useState([]);
  const [presets, setPresets] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch all available templates (both user and presets)
   */
  const fetchTemplates = useCallback(async (includePresets = true) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/templates?include_presets=${includePresets}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch templates");
      }
      const data = await response.json();

      // Separate presets from user templates
      const builtinPresets = data.templates.filter((t) => t.is_builtin);
      const userTemplates = data.templates.filter((t) => !t.is_builtin);

      setPresets(builtinPresets);
      setTemplates(userTemplates);

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Fetch only presets
   */
  const fetchPresets = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/templates/presets`);
      if (!response.ok) {
        throw new Error("Failed to fetch presets");
      }
      const data = await response.json();
      setPresets(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Load a specific template by name
   */
  const loadTemplate = useCallback(async (name) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/templates/${encodeURIComponent(name)}`
      );
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`Template "${name}" not found`);
        }
        throw new Error("Failed to load template");
      }
      return await response.json();
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Save a template
   */
  const saveTemplate = useCallback(async (template, overwrite = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/templates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template, overwrite }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to save template");
      }
      const result = await response.json();
      // Refresh templates list
      await fetchTemplates();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchTemplates]);

  /**
   * Delete a template
   */
  const deleteTemplate = useCallback(async (name) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/templates/${encodeURIComponent(name)}`,
        { method: "DELETE" }
      );
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to delete template");
      }
      // Refresh templates list
      await fetchTemplates();
      return true;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchTemplates]);

  /**
   * Export a template as YAML
   */
  const exportTemplate = useCallback(async (name) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/templates/export?name=${encodeURIComponent(name)}`,
        { method: "POST" }
      );
      if (!response.ok) {
        throw new Error("Failed to export template");
      }
      const data = await response.json();

      // Create download
      const blob = new Blob([data.yaml_content], { type: "application/x-yaml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Import a template from YAML content
   */
  const importTemplate = useCallback(async (yamlContent, nameOverride = null) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/templates/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          yaml_content: yamlContent,
          name: nameOverride,
        }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to import template");
      }
      const result = await response.json();
      // Refresh templates list
      await fetchTemplates();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchTemplates]);

  /**
   * Import a template from a file
   */
  const importTemplateFromFile = useCallback(async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const result = await importTemplate(e.target.result);
          resolve(result);
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsText(file);
    });
  }, [importTemplate]);

  return {
    templates,
    presets,
    isLoading,
    error,
    fetchTemplates,
    fetchPresets,
    loadTemplate,
    saveTemplate,
    deleteTemplate,
    exportTemplate,
    importTemplate,
    importTemplateFromFile,
  };
}

export default useTemplates;
