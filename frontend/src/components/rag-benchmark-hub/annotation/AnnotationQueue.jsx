import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Badge } from "../../ui/badge";
import { ScrollArea } from "../../ui/scroll-area";
import { CheckCircle, Circle, XCircle } from "lucide-react";
import { cn } from "../../../lib/utils";

/**
 * Queue view showing all pairs with their annotation status
 */
export function AnnotationQueue({ pairs, annotations, currentIndex, onSelectIndex }) {
  const getStatusIcon = (status) => {
    switch (status) {
      case "approved":
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "rejected":
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Circle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Annotation Queue</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[400px]">
          <div className="space-y-1 p-3">
            {pairs.map((pair, idx) => {
              const annotation = annotations[idx] || { status: "pending" };
              const isCurrent = idx === currentIndex;

              return (
                <div
                  key={idx}
                  className={cn(
                    "flex items-center gap-2 p-2 rounded-md cursor-pointer transition-colors",
                    isCurrent
                      ? "bg-primary/10 border border-primary/30"
                      : "hover:bg-muted/50"
                  )}
                  onClick={() => onSelectIndex(idx)}
                >
                  {getStatusIcon(annotation.status)}
                  <span className="flex-1 text-sm truncate">
                    {pair.query || pair.question}
                  </span>
                  <Badge
                    variant="outline"
                    className="text-[10px] px-1 py-0"
                  >
                    #{idx + 1}
                  </Badge>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

AnnotationQueue.propTypes = {
  pairs: PropTypes.array.isRequired,
  annotations: PropTypes.object.isRequired,
  currentIndex: PropTypes.number.isRequired,
  onSelectIndex: PropTypes.func.isRequired,
};

export default AnnotationQueue;
