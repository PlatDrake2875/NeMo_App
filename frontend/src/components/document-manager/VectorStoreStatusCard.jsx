import PropTypes from "prop-types";
import { Database, RefreshCw } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";

export function VectorStoreStatusCard({
  backend,
  connectionStatus,
  embeddingModel,
  onRefresh,
  isLoading
}) {
  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected":
        return "bg-green-500";
      case "disconnected":
        return "bg-red-500";
      case "checking":
        return "bg-yellow-500 animate-pulse";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusBadge = () => {
    switch (connectionStatus) {
      case "connected":
        return <Badge variant="outline" className="border-green-500 text-green-600">Connected</Badge>;
      case "disconnected":
        return <Badge variant="outline" className="border-red-500 text-red-600">Disconnected</Badge>;
      case "checking":
        return <Badge variant="outline" className="border-yellow-500 text-yellow-600">Checking...</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  const getBackendDisplayName = () => {
    switch (backend?.toLowerCase()) {
      case "pgvector":
        return "PostgreSQL + pgvector";
      case "qdrant":
        return "Qdrant";
      case "chroma":
        return "ChromaDB";
      default:
        return backend || "Unknown";
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Vector Store</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={isLoading}
            className="h-8 w-8"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
        </div>
        <CardDescription>
          Current vector database configuration and status
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Status</span>
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${getStatusColor()}`} />
            {getStatusBadge()}
          </div>
        </div>

        {/* Backend Type */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Backend</span>
          <span className="text-sm font-medium">{getBackendDisplayName()}</span>
        </div>

        {/* Embedding Model */}
        {embeddingModel && (
          <div className="flex flex-col gap-1">
            <span className="text-sm text-muted-foreground">Embedding Model</span>
            <code className="text-xs bg-muted px-2 py-1 rounded break-all">
              {embeddingModel}
            </code>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

VectorStoreStatusCard.propTypes = {
  backend: PropTypes.string,
  connectionStatus: PropTypes.oneOf(["connected", "disconnected", "checking", "unknown"]),
  embeddingModel: PropTypes.string,
  onRefresh: PropTypes.func.isRequired,
  isLoading: PropTypes.bool
};
