/**
 * API Configuration Utility
 *
 * Centralized API base URL configuration with localStorage support for runtime configuration.
 * Priority: URL param (?backend=) > localStorage > environment variable > localhost fallback
 */

const BACKEND_URL_STORAGE_KEY = "nemo_backend_url";
const DEFAULT_BACKEND_URL = "http://localhost:8000";

/**
 * Get the backend API base URL.
 * Checks URL param, then localStorage, then environment variable, then falls back to localhost.
 */
export const getApiBaseUrl = () => {
  // First check URL parameter (allows setting via ?backend=https://example.com)
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const backendParam = urlParams.get("backend");
    if (backendParam) {
      // Save to localStorage for future use
      localStorage.setItem(BACKEND_URL_STORAGE_KEY, JSON.stringify(backendParam));
      console.log("[API Config] Backend URL set from URL parameter:", backendParam);
      return backendParam;
    }
  } catch (e) {
    // URL parsing failed, continue to other methods
  }

  // Then check localStorage for user-configured URL
  try {
    const stored = localStorage.getItem(BACKEND_URL_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed && typeof parsed === "string") {
        return parsed;
      }
    }
  } catch (e) {
    console.warn("[API Config] Failed to read backend URL from localStorage:", e);
  }

  // Fall back to environment variable
  const envUrl = import.meta.env.VITE_API_URL;

  if (!envUrl) {
    if (import.meta.env.PROD) {
      console.warn(
        "[API Config] VITE_API_URL environment variable is not set. " +
          "Using localhost fallback. This is likely a configuration error in production."
      );
    }
    return DEFAULT_BACKEND_URL;
  }

  return envUrl;
};

/**
 * Set the backend API URL in localStorage.
 */
export const setApiBaseUrl = (url) => {
  const normalized = url.replace(/\/+$/, ""); // Remove trailing slashes
  localStorage.setItem(BACKEND_URL_STORAGE_KEY, JSON.stringify(normalized));
};

/**
 * Get default URLs (for reset functionality).
 */
export const getDefaultUrls = () => ({
  backend: import.meta.env.VITE_API_URL || DEFAULT_BACKEND_URL,
});

/**
 * The base URL for API requests.
 * Note: This is evaluated once at module load time.
 * For dynamic updates, use getApiBaseUrl() function.
 */
export const API_BASE_URL = getApiBaseUrl();
