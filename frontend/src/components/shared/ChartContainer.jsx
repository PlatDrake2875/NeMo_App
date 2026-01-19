import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/card";
import { cn } from "../../lib/utils";

/**
 * Wrapper component for charts with consistent styling
 */
export function ChartContainer({
  title,
  description,
  children,
  className,
  headerClassName,
  contentClassName,
  actions,
}) {
  return (
    <Card className={cn("", className)}>
      {(title || description || actions) && (
        <CardHeader className={cn("pb-2", headerClassName)}>
          <div className="flex items-center justify-between">
            <div>
              {title && <CardTitle className="text-base">{title}</CardTitle>}
              {description && (
                <CardDescription className="mt-1">{description}</CardDescription>
              )}
            </div>
            {actions && <div className="flex items-center gap-2">{actions}</div>}
          </div>
        </CardHeader>
      )}
      <CardContent className={cn("pt-0", contentClassName)}>
        {children}
      </CardContent>
    </Card>
  );
}

ChartContainer.propTypes = {
  title: PropTypes.string,
  description: PropTypes.string,
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  headerClassName: PropTypes.string,
  contentClassName: PropTypes.string,
  actions: PropTypes.node,
};

export default ChartContainer;
