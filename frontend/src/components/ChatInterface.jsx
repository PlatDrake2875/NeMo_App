import { ChatForm } from "./ChatForm";
import { ChatHistory } from "./ChatHistory";
import { Header } from "./Header";
import { MessageCircle } from "lucide-react";

export function ChatInterface({
  activeSessionId,
  activeSessionName,
  chatHistory,
  onSubmit,
  onClearHistory,
  onDownloadHistory,
  isSubmitting,
  onStopGeneration,
  isDarkMode,
  toggleTheme,
  availableModels,
  selectedModel,
  onModelChange,
  modelsLoading,
  modelsError,
  isInitialized,
  sessionAgents,
  onRefreshModels,
}) {
  // Determine general disabled state
  const isDisabled =
    !isInitialized ||
    !activeSessionId ||
    modelsLoading ||
    !!modelsError ||
    isSubmitting;

  // Determine if form specifically should be disabled
  const isFormDisabled = isDisabled || !selectedModel;

  // Check if agent is selected for this session
  const hasAgent = sessionAgents[activeSessionId] !== undefined;

  return (
    <main className="flex flex-col h-full bg-background">
      <Header
        activeSessionName={activeSessionName}
        chatHistory={chatHistory}
        clearChatHistory={onClearHistory}
        downloadChatHistory={onDownloadHistory}
        disabled={isDisabled}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
        availableModels={availableModels}
        selectedModel={selectedModel}
        onModelChange={onModelChange}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
        onRefreshModels={onRefreshModels}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {!isInitialized ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-3">
              <div className="animate-pulse">
                <MessageCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              </div>
              <p className="text-muted-foreground">Loading sessions...</p>
            </div>
          </div>
        ) : activeSessionId ? (
          chatHistory.length > 0 || hasAgent ? (
            <ChatHistory chatHistory={chatHistory} isLoading={isSubmitting} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-3 max-w-md px-6">
                <MessageCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground">
                  Start Chatting
                </h3>
                <p className="text-sm text-muted-foreground">
                  Type a message below to start the conversation. You can select an AI assistant or skip to continue.
                </p>
              </div>
            </div>
          )
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-3 max-w-md px-6">
              <MessageCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground">
                No Chat Selected
              </h3>
              <p className="text-sm text-muted-foreground">
                Select a chat from the sidebar or start a new one to begin.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Input Area - Only show when ready */}
      {isInitialized && activeSessionId && (
        <ChatForm
          onSubmit={onSubmit}
          disabled={isFormDisabled}
          isSubmitting={isSubmitting}
          onStopGeneration={onStopGeneration}
        />
      )}
    </main>
  );
}
