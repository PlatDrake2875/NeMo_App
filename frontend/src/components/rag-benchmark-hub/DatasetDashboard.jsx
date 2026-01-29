import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import {
  Database,
  FolderOpen,
  HardDrive,
  Loader2,
  Plus,
  RefreshCw,
} from "lucide-react";
import { getApiBaseUrl } from "../../lib/api-config";

export function DatasetDashboard({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/processed-datasets/stats`);
      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.status}`);
      }
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error("Error fetching stats:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-destructive">Error: {error}</p>
        <Button onClick={fetchStats} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Quick Actions */}
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => onNavigate("raw")}>
          <Plus className="h-4 w-4 mr-2" />
          Create Raw Dataset
        </Button>
        <Button variant="outline" onClick={() => onNavigate("huggingface")}>
          Import from HuggingFace
        </Button>
        <Button variant="ghost" size="icon" onClick={fetchStats}>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card
          className="cursor-pointer hover:bg-accent/50 transition-colors"
          onClick={() => onNavigate("raw")}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Raw Datasets</CardTitle>
            <FolderOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.total_raw_datasets ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats?.total_raw_files ?? 0} files total
            </p>
          </CardContent>
        </Card>

        <Card
          className="cursor-pointer hover:bg-accent/50 transition-colors"
          onClick={() => onNavigate("processed")}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Processed Datasets
            </CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.total_processed_datasets ?? 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats?.processing_in_progress ?? 0} processing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBytes(stats?.total_storage_bytes ?? 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              in raw datasets
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Datasets by Backend */}
      {stats?.datasets_by_backend && Object.keys(stats.datasets_by_backend).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Datasets by Vector Backend</CardTitle>
            <CardDescription>Distribution of processed datasets</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              {Object.entries(stats.datasets_by_backend).map(([backend, count]) => (
                <div
                  key={backend}
                  className="flex items-center gap-2 bg-muted px-3 py-2 rounded-md"
                >
                  <span className="font-medium capitalize">{backend}</span>
                  <span className="text-muted-foreground">({count})</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Datasets by Chunking Method */}
      {stats?.datasets_by_chunking_method && Object.keys(stats.datasets_by_chunking_method).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Datasets by Chunking Method</CardTitle>
            <CardDescription>Distribution of chunking strategies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              {Object.entries(stats.datasets_by_chunking_method).map(
                ([method, count]) => (
                  <div
                    key={method}
                    className="flex items-center gap-2 bg-muted px-3 py-2 rounded-md"
                  >
                    <span className="font-medium capitalize">{method}</span>
                    <span className="text-muted-foreground">({count})</span>
                  </div>
                )
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Empty State */}
      {stats?.total_raw_datasets === 0 && stats?.total_processed_datasets === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <FolderOpen className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No datasets yet</h3>
            <p className="text-muted-foreground mb-4">
              Get started by creating a raw dataset or importing from HuggingFace
            </p>
            <div className="flex gap-2">
              <Button onClick={() => onNavigate("raw")}>
                <Plus className="h-4 w-4 mr-2" />
                Create Raw Dataset
              </Button>
              <Button variant="outline" onClick={() => onNavigate("huggingface")}>
                Import from HuggingFace
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

DatasetDashboard.propTypes = {
  onNavigate: PropTypes.func.isRequired,
};

export default DatasetDashboard;
