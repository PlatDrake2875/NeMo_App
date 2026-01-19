import PropTypes from "prop-types";
import { Card, CardContent } from "../ui/card";
import { cn } from "../../lib/utils";

/**
 * Dashboard stat card component
 */
export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  className,
  valueClassName,
}) {
  const getTrendColor = () => {
    if (!trend) return "";
    if (trend > 0) return "text-green-600";
    if (trend < 0) return "text-red-600";
    return "text-muted-foreground";
  };

  const formatTrend = () => {
    if (!trend) return null;
    const sign = trend > 0 ? "+" : "";
    return `${sign}${trend.toFixed(1)}%`;
  };

  return (
    <Card className={cn("", className)}>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <div className="flex items-baseline gap-2">
              <p className={cn("text-2xl font-bold", valueClassName)}>{value}</p>
              {trend !== undefined && trend !== null && (
                <span className={cn("text-sm font-medium", getTrendColor())}>
                  {formatTrend()}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {Icon && (
            <div className="p-2 bg-muted rounded-md">
              <Icon className="h-5 w-5 text-muted-foreground" />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

StatCard.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  subtitle: PropTypes.string,
  icon: PropTypes.elementType,
  trend: PropTypes.number,
  className: PropTypes.string,
  valueClassName: PropTypes.string,
};

export default StatCard;
