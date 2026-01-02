/**
 * API Configuration Utility
 *
 * Centralized API base URL configuration with production warnings.
 * This prevents hardcoded localhost URLs from causing issues in production.
 */

const getApiBaseUrl = () => {
  const envUrl = import.meta.env.VITE_API_URL;

  if (!envUrl) {
    // In production builds, warn about missing configuration
    if (import.meta.env.PROD) {
      console.warn(
        "[API Config] VITE_API_URL environment variable is not set. " +
          "Using localhost fallback. This is likely a configuration error in production."
      );
    }
    return "http://localhost:8000";
  }

  return envUrl;
};

/**
 * The base URL for API requests.
 * In development, defaults to http://localhost:8000 if VITE_API_URL is not set.
 * In production, logs a warning if VITE_API_URL is missing.
 */
export const API_BASE_URL = getApiBaseUrl();
