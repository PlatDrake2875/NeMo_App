// frontend/src/components/guardrails/GuardrailsEditor.jsx
import PropTypes from "prop-types";
import { useState } from "react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Alert } from "../ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  ArrowLeft,
  Save,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Loader2,
  Shield,
  FileCode,
  Play,
  Plus,
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useGuardrailsEditor } from "../../hooks/useGuardrailsEditor";
import { AgentList } from "./AgentList";
import { ConfigEditor } from "./ConfigEditor";
import { GuardrailsTester } from "./GuardrailsTester";

/**
 * GuardrailsEditor - Main editor for NeMo Guardrails configurations
 */
export function GuardrailsEditor({ onBack, className }) {
  const {
    agents,
    selectedAgent,
    isLoading,
    isSaving,
    error,
    saveError,
    configYaml,
    setConfigYaml,
    configColang,
    setConfigColang,
    validationResult,
    isValidating,
    validateConfig,
    loadAgentConfig,
    saveAgentConfig,
    createAgent,
    cloneAgent,
    deleteAgent,
    hasUnsavedChanges,
    resetChanges,
  } = useGuardrailsEditor();

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showCloneDialog, setShowCloneDialog] = useState(false);
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentDescription, setNewAgentDescription] = useState("");
  const [cloneSourceAgent, setCloneSourceAgent] = useState("");
  const [activeTab, setActiveTab] = useState("yaml");

  // Handle create agent
  const handleCreate = async () => {
    const success = await createAgent(newAgentName, newAgentDescription);
    if (success) {
      setShowCreateDialog(false);
      setNewAgentName("");
      setNewAgentDescription("");
    }
  };

  // Handle clone agent
  const handleClone = async () => {
    const success = await cloneAgent(cloneSourceAgent, newAgentName);
    if (success) {
      setShowCloneDialog(false);
      setCloneSourceAgent("");
      setNewAgentName("");
    }
  };

  // Handle save with validation
  const handleSave = async () => {
    const validation = await validateConfig();
    if (validation?.valid) {
      await saveAgentConfig();
    }
  };

  return (
    <div className={cn("flex h-full bg-background", className)}>
      {/* Left Panel - Agent List */}
      <div className="w-72 border-r flex-shrink-0">
        <AgentList
          agents={agents}
          selectedAgent={selectedAgent}
          onSelectAgent={loadAgentConfig}
          onCreateAgent={() => setShowCreateDialog(true)}
          onCloneAgent={(name) => {
            setCloneSourceAgent(name);
            setNewAgentName(`${name}-copy`);
            setShowCloneDialog(true);
          }}
          onDeleteAgent={deleteAgent}
          isLoading={isLoading}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="border-b px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={onBack}>
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-semibold">
                {selectedAgent ? selectedAgent.name : "Guardrails Editor"}
              </h1>
              {selectedAgent?.is_custom && (
                <Badge variant="outline">Custom</Badge>
              )}
              {hasUnsavedChanges && (
                <Badge variant="secondary">Unsaved</Badge>
              )}
            </div>
          </div>

          {selectedAgent && (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={validateConfig}
                disabled={isValidating}
              >
                {isValidating ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <CheckCircle className="h-4 w-4 mr-1" />
                )}
                Validate
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={resetChanges}
                disabled={!hasUnsavedChanges || isSaving}
              >
                <RotateCcw className="h-4 w-4 mr-1" />
                Reset
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={!hasUnsavedChanges || isSaving || !selectedAgent?.is_custom}
              >
                {isSaving ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-1" />
                )}
                Save
              </Button>
            </div>
          )}
        </div>

        {/* Error/Validation Messages */}
        {(error || saveError || validationResult) && (
          <div className="px-4 py-2 space-y-2">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <span className="ml-2 text-sm">{error}</span>
              </Alert>
            )}
            {saveError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <span className="ml-2 text-sm">Save failed: {saveError}</span>
              </Alert>
            )}
            {validationResult && !validationResult.valid && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <div className="ml-2">
                  <span className="text-sm font-medium">Validation errors:</span>
                  <ul className="text-xs mt-1 list-disc list-inside">
                    {validationResult.errors?.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              </Alert>
            )}
            {validationResult?.valid && (
              <Alert className="border-green-500 text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span className="ml-2 text-sm">Configuration is valid</span>
              </Alert>
            )}
          </div>
        )}

        {/* Content Area */}
        {selectedAgent ? (
          <div className="flex-1 flex overflow-hidden">
            {/* Editor Panel */}
            <div className="flex-1 flex flex-col min-w-0">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
                <div className="border-b px-4">
                  <TabsList>
                    <TabsTrigger value="yaml" className="flex items-center gap-1">
                      <FileCode className="h-4 w-4" />
                      config.yml
                    </TabsTrigger>
                    <TabsTrigger value="colang" className="flex items-center gap-1">
                      <FileCode className="h-4 w-4" />
                      config.co
                    </TabsTrigger>
                  </TabsList>
                </div>
                <TabsContent value="yaml" className="flex-1 p-4 mt-0">
                  <ConfigEditor
                    value={configYaml}
                    onChange={setConfigYaml}
                    language="yaml"
                    readOnly={!selectedAgent?.is_custom}
                    height="100%"
                  />
                </TabsContent>
                <TabsContent value="colang" className="flex-1 p-4 mt-0">
                  <ConfigEditor
                    value={configColang}
                    onChange={setConfigColang}
                    language="colang"
                    readOnly={!selectedAgent?.is_custom}
                    height="100%"
                  />
                </TabsContent>
              </Tabs>
            </div>

            {/* Test Panel */}
            <div className="w-80 border-l flex-shrink-0">
              <div className="h-full flex flex-col">
                <div className="border-b px-4 py-3 flex items-center gap-2">
                  <Play className="h-4 w-4 text-primary" />
                  <span className="font-medium text-sm">Test Guardrails</span>
                </div>
                <GuardrailsTester agentName={selectedAgent?.name} className="flex-1" />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-4">
              <Shield className="h-16 w-16 text-muted-foreground mx-auto" />
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">No Agent Selected</h3>
                <p className="text-sm text-muted-foreground max-w-sm">
                  Select an agent from the list to view and edit its configuration,
                  or create a new custom agent.
                </p>
              </div>
              <Button onClick={() => setShowCreateDialog(true)}>
                <Plus className="h-4 w-4 mr-1" />
                Create New Agent
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Create Agent Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Agent</DialogTitle>
            <DialogDescription>
              Create a new custom guardrails agent with default configuration.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="agent-name">Agent Name</Label>
              <Input
                id="agent-name"
                value={newAgentName}
                onChange={(e) => setNewAgentName(e.target.value)}
                placeholder="my-custom-agent"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="agent-description">Description (optional)</Label>
              <Input
                id="agent-description"
                value={newAgentDescription}
                onChange={(e) => setNewAgentDescription(e.target.value)}
                placeholder="A custom agent for..."
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={!newAgentName.trim()}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Clone Agent Dialog */}
      <Dialog open={showCloneDialog} onOpenChange={setShowCloneDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clone Agent</DialogTitle>
            <DialogDescription>
              Create a copy of "{cloneSourceAgent}" with a new name.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="clone-name">New Agent Name</Label>
              <Input
                id="clone-name"
                value={newAgentName}
                onChange={(e) => setNewAgentName(e.target.value)}
                placeholder="my-cloned-agent"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCloneDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleClone} disabled={!newAgentName.trim()}>
              Clone
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

GuardrailsEditor.propTypes = {
  onBack: PropTypes.func.isRequired,
  className: PropTypes.string,
};
