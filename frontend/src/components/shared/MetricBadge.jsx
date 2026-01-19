import PropTypes from "prop-types";
import { Badge } from "../ui/badge";
import { cn } from "../../lib/utils";

/**
 * Color-coded badge for displaying metric scores
 */
export function MetricBadge({
  value,
  label,
  showPercentage = true,
  size = "default",
  className,
}) {
  const percentage = value * 100;

  // Determine color based on score
  const getVariant = (score) => {
    if (score >= 80) return "default"; // Green-ish
    if (score >= 50) return "secondary"; // Yellow-ish
    return "destructive"; // Red
  };

  const getColorClass = (score) => {
    if (score >= 80) return "bg-green-500/10 text-green-600 border-green-500/30";
    if (score >= 50) return "bg-yellow-500/10 text-yellow-600 border-yellow-500/30";
    return "bg-red-500/10 text-red-600 border-red-500/30";
  };

  const sizeClass = size === "sm" ? "text-xs px-1.5 py-0.5" : "text-sm px-2 py-0.5";

  return (
    <Badge
      variant="outline"
      className={cn(getColorClass(percentage), sizeClass, className)}
    >
      {label && <span className="mr-1 opacity-70">{label}:</span>}
      {showPercentage ? `${percentage.toFixed(0)}%` : value.toFixed(2)}
    </Badge>
  );
}

MetricBadge.propTypes = {
  value: PropTypes.number.isRequired,
  label: PropTypes.string,
  showPercentage: PropTypes.bool,
  size: PropTypes.oneOf(["sm", "default"]),
  className: PropTypes.string,
};

export default MetricBadge;
