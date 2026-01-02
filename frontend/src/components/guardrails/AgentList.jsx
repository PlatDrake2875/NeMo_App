// frontend/src/components/guardrails/AgentList.jsx
import PropTypes from "prop-types";
import { useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { Skeleton } from "../ui/skeleton";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../ui/alert-dialog";
import {
  Plus,
  Search,
  Shield,
  Trash2,
  Copy,
  ChevronRight,
} from "lucide-react";
import { cn } from "../../lib/utils";

/**
 * AgentList - Left panel showing list of available agents
 */
export function AgentList({
  agents,
  selectedAgent,
  onSelectAgent,
  onCreateAgent,
  onCloneAgent,
  onDeleteAgent,
  isLoading,
  className,
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteTarget, setDeleteTarget] = useState(null);

  const filteredAgents = agents.filter((agent) =>
    agent.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleDelete = async () => {
    if (deleteTarget) {
      await onDeleteAgent(deleteTarget.name);
      setDeleteTarget(null);
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-sm">Guardrails Agents</h2>
          <Button
            size="sm"
            onClick={onCreateAgent}
            disabled={isLoading}
          >
            <Plus className="h-4 w-4 mr-1" />
            New
          </Button>
        </div>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8 h-8 text-sm"
          />
        </div>
      </div>

      {/* Agent List */}
      <div className="flex-1 overflow-y-auto p-2">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} className="h-16 rounded-lg" />
            ))}
          </div>
        ) : filteredAgents.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            {searchQuery ? "No agents match your search" : "No agents available"}
          </div>
        ) : (
          <div className="space-y-1">
            {filteredAgents.map((agent) => (
              <div
                key={agent.name}
                className={cn(
                  "group relative rounded-lg border p-3 cursor-pointer transition-colors",
                  "hover:bg-accent",
                  selectedAgent?.name === agent.name && "bg-accent border-primary"
                )}
                onClick={() => onSelectAgent(agent.name)}
              >
                <div className="flex items-start gap-3">
                  <Shield className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm truncate">
                        {agent.name}
                      </span>
                      {agent.is_custom && (
                        <Badge variant="outline" className="text-xs">
                          Custom
                        </Badge>
                      )}
                    </div>
                    {agent.description && (
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {agent.description}
                      </p>
                    )}
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>

                {/* Action Buttons */}
                <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      onCloneAgent(agent.name);
                    }}
                    title="Clone agent"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                  {agent.is_custom && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 hover:bg-destructive/10 hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget(agent);
                      }}
                      title="Delete agent"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Agent</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deleteTarget?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

AgentList.propTypes = {
  agents: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      description: PropTypes.string,
      is_custom: PropTypes.bool,
    })
  ).isRequired,
  selectedAgent: PropTypes.shape({
    name: PropTypes.string,
  }),
  onSelectAgent: PropTypes.func.isRequired,
  onCreateAgent: PropTypes.func.isRequired,
  onCloneAgent: PropTypes.func.isRequired,
  onDeleteAgent: PropTypes.func.isRequired,
  isLoading: PropTypes.bool,
  className: PropTypes.string,
};
