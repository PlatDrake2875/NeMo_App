import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Loader2, ChevronRight, AlertCircle } from "lucide-react";
import { cn } from "../lib/utils";

export function AgentSelector({ onAgentSelect, onCancel }) {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await fetch("/api/agents/metadata");
        if (!response.ok) {
          throw new Error("Failed to fetch agent metadata");
        }
        const data = await response.json();
        setAgents(data.agents || []);
      } catch (err) {
        console.error("Error fetching agents:", err);
        setError("Failed to load available agents");
        // Fallback to default agents
        setAgents([
          {
            name: "Math Assistant",
            directory: "math_assistant",
            description:
              "Specialized in mathematics, equations, and mathematical concepts",
            icon: "🧮",
            persona: "Martin Scorsese-inspired math specialist",
          },
          {
            name: "Bank Assistant",
            directory: "bank_assistant",
            description:
              "Expert in banking, financial services, and account management",
            icon: "🏦",
            persona: "Professional banking advisor",
          },
          {
            name: "Aviation Assistant",
            directory: "aviation_assistant",
            description:
              "Specialist in flight operations, aircraft systems, and aviation",
            icon: "✈️",
            persona: "Aviation operations expert",
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  const handleAgentSelect = (agent) => {
    onAgentSelect(agent.directory);
  };

  return (
    <Dialog open={true} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Choose Your AI Assistant</DialogTitle>
          <DialogDescription>
            Select a specialized assistant for your conversation
          </DialogDescription>
          {error && (
            <div className="flex items-center gap-2 text-sm text-amber-600 dark:text-amber-400 mt-2">
              <AlertCircle className="h-4 w-4" />
              <span>{error} (using defaults)</span>
            </div>
          )}
        </DialogHeader>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading assistants...</p>
          </div>
        ) : error && agents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <AlertCircle className="h-12 w-12 text-destructive" />
            <p className="text-sm text-destructive">{error}</p>
          </div>
        ) : (
          <div className="grid gap-3 py-4 max-h-[60vh] overflow-y-auto scrollbar-thin">
            {agents.map((agent, index) => (
              <Card
                key={agent.directory || index}
                className="cursor-pointer hover:border-primary transition-all hover:shadow-md group"
                onClick={() => handleAgentSelect(agent)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="text-4xl flex-shrink-0">{agent.icon}</div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-base mb-1">
                        {agent.name}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-1">
                        {agent.description}
                      </p>
                      <p className="text-xs text-muted-foreground/80 italic">
                        {agent.persona}
                      </p>
                    </div>
                    <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all flex-shrink-0" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <DialogFooter className="flex-row justify-between sm:justify-between">
          <Button
            variant="ghost"
            onClick={() => onAgentSelect(null)}
            className="text-muted-foreground"
          >
            Continue Without Assistant
          </Button>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
