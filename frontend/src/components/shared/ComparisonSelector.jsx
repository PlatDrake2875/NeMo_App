import { useState } from "react";
import PropTypes from "prop-types";
import { Check, GitCompare } from "lucide-react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { cn } from "../../lib/utils";

/**
 * Multi-select component for selecting items to compare
 */
export function ComparisonSelector({
  items,
  selectedIds,
  onSelectionChange,
  maxSelections = 5,
  renderItem,
  className,
}) {
  const handleToggle = (id) => {
    if (selectedIds.includes(id)) {
      onSelectionChange(selectedIds.filter((i) => i !== id));
    } else if (selectedIds.length < maxSelections) {
      onSelectionChange([...selectedIds, id]);
    }
  };

  const handleClear = () => {
    onSelectionChange([]);
  };

  return (
    <div className={cn("space-y-3", className)}>
      {/* Selection header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitCompare className="h-4 w-4" />
          <span className="text-sm font-medium">
            Select items to compare ({selectedIds.length}/{maxSelections})
          </span>
        </div>
        {selectedIds.length > 0 && (
          <Button variant="ghost" size="sm" onClick={handleClear}>
            Clear selection
          </Button>
        )}
      </div>

      {/* Selected badges */}
      {selectedIds.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selectedIds.map((id, index) => {
            const item = items.find((i) => i.id === id);
            return (
              <Badge
                key={id}
                variant="secondary"
                className="cursor-pointer"
                onClick={() => handleToggle(id)}
              >
                <span className="mr-1 opacity-50">#{index + 1}</span>
                {item?.name || id}
                <span className="ml-1 opacity-50">×</span>
              </Badge>
            );
          })}
        </div>
      )}

      {/* Items list */}
      <div className="space-y-1.5 max-h-80 overflow-y-auto">
        {items.map((item) => {
          const isSelected = selectedIds.includes(item.id);
          const canSelect = selectedIds.length < maxSelections || isSelected;

          return (
            <div
              key={item.id}
              className={cn(
                "flex items-center gap-2 p-2 rounded-md border cursor-pointer transition-colors",
                isSelected
                  ? "border-primary bg-primary/5"
                  : canSelect
                  ? "border-transparent hover:border-muted-foreground/30 hover:bg-muted/50"
                  : "border-transparent opacity-50 cursor-not-allowed"
              )}
              onClick={() => canSelect && handleToggle(item.id)}
            >
              {/* Selection indicator */}
              <div
                className={cn(
                  "h-5 w-5 rounded border flex items-center justify-center flex-shrink-0",
                  isSelected
                    ? "bg-primary border-primary text-primary-foreground"
                    : "border-muted-foreground/30"
                )}
              >
                {isSelected && <Check className="h-3 w-3" />}
              </div>

              {/* Item content */}
              <div className="flex-1 min-w-0">
                {renderItem ? (
                  renderItem(item)
                ) : (
                  <span className="truncate">{item.name || item.id}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

ComparisonSelector.propTypes = {
  items: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string,
    })
  ).isRequired,
  selectedIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  onSelectionChange: PropTypes.func.isRequired,
  maxSelections: PropTypes.number,
  renderItem: PropTypes.func,
  className: PropTypes.string,
};

export default ComparisonSelector;
