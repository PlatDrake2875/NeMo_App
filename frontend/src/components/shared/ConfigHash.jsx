import { useState } from "react";
import PropTypes from "prop-types";
import { Check, Copy, Hash } from "lucide-react";
import { Button } from "../ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { cn } from "../../lib/utils";

/**
 * Clickable config hash with copy functionality
 */
export function ConfigHash({ hash, showIcon = true, className }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(hash);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className={cn(
              "h-auto py-0.5 px-1.5 font-mono text-xs text-muted-foreground hover:text-foreground",
              className
            )}
            onClick={handleCopy}
          >
            {showIcon && (
              copied ? (
                <Check className="h-3 w-3 mr-1 text-green-500" />
              ) : (
                <Hash className="h-3 w-3 mr-1" />
              )
            )}
            {hash}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{copied ? "Copied!" : "Click to copy config hash"}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

ConfigHash.propTypes = {
  hash: PropTypes.string.isRequired,
  showIcon: PropTypes.bool,
  className: PropTypes.string,
};

export default ConfigHash;
