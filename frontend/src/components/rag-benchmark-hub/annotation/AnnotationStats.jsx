import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { CheckCircle, Circle, XCircle } from "lucide-react";

/**
 * Statistics card for annotation progress
 */
export function AnnotationStats({ stats }) {
  const completionRate =
    stats.total > 0
      ? (((stats.approved + stats.rejected) / stats.total) * 100).toFixed(1)
      : 0;

  const approvalRate =
    stats.approved + stats.rejected > 0
      ? ((stats.approved / (stats.approved + stats.rejected)) * 100).toFixed(1)
      : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Annotation Statistics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress stats */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              Approved
            </span>
            <span className="font-medium">{stats.approved}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm flex items-center gap-2">
              <XCircle className="h-4 w-4 text-red-600" />
              Rejected
            </span>
            <span className="font-medium">{stats.rejected}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm flex items-center gap-2">
              <Circle className="h-4 w-4 text-muted-foreground" />
              Pending
            </span>
            <span className="font-medium">{stats.pending}</span>
          </div>
        </div>

        <hr />

        {/* Summary metrics */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Total Pairs</span>
            <span className="font-medium">{stats.total}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Completion Rate</span>
            <span className="font-medium">{completionRate}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Approval Rate</span>
            <span className="font-medium text-green-600">{approvalRate}%</span>
          </div>
        </div>

        {/* Visual bar */}
        <div className="h-4 rounded-full overflow-hidden bg-muted flex">
          <div
            className="bg-green-500 transition-all"
            style={{
              width: `${stats.total > 0 ? (stats.approved / stats.total) * 100 : 0}%`,
            }}
          />
          <div
            className="bg-red-500 transition-all"
            style={{
              width: `${stats.total > 0 ? (stats.rejected / stats.total) * 100 : 0}%`,
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}

AnnotationStats.propTypes = {
  stats: PropTypes.shape({
    total: PropTypes.number.isRequired,
    approved: PropTypes.number.isRequired,
    rejected: PropTypes.number.isRequired,
    pending: PropTypes.number.isRequired,
  }).isRequired,
};

export default AnnotationStats;
