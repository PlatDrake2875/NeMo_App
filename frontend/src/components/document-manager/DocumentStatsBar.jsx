import PropTypes from "prop-types";
import { FileText, Layers, RefreshCw } from "lucide-react";
import { Button } from "../ui/button";

export function DocumentStatsBar({
  stats,
  isLoading,
  onRefresh
}) {
  return (
    <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
      <div className="flex items-center gap-6">
        {/* Total Documents */}
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-2xl font-bold">
              {isLoading ? "-" : (stats?.total_documents || 0)}
            </p>
            <p className="text-xs text-muted-foreground">Documents</p>
          </div>
        </div>

        {/* Total Chunks */}
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-2xl font-bold">
              {isLoading ? "-" : (stats?.total_chunks || 0)}
            </p>
            <p className="text-xs text-muted-foreground">Chunks</p>
          </div>
        </div>

        {/* Average Chunks per Doc */}
        {stats?.total_documents > 0 && (
          <div className="hidden sm:block">
            <p className="text-lg font-semibold">
              {Math.round(stats.total_chunks / stats.total_documents)}
            </p>
            <p className="text-xs text-muted-foreground">Avg chunks/doc</p>
          </div>
        )}
      </div>

      {/* Refresh Button */}
      <Button
        variant="outline"
        size="sm"
        onClick={onRefresh}
        disabled={isLoading}
      >
        <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
        Refresh
      </Button>
    </div>
  );
}

DocumentStatsBar.propTypes = {
  stats: PropTypes.shape({
    total_documents: PropTypes.number,
    total_chunks: PropTypes.number
  }),
  isLoading: PropTypes.bool,
  onRefresh: PropTypes.func.isRequired
};
