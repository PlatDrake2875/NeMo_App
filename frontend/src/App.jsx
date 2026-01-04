// HIA/frontend/src/App.jsx
import { useCallback, useEffect, useMemo, useState } from "react";
import "./index.css";

import { AgentSelector } from "./components/AgentSelector";
import { ChatInterface } from "./components/ChatInterface";
import { RAGBenchmarkHub } from "./components/RAGBenchmarkHub";
import { GuardrailsEditor } from "./components/guardrails";

// Import components
import { Sidebar } from "./components/Sidebar";
import { TooltipProvider } from "./components/ui/tooltip";
import { useChatSessions } from "./hooks/useChatSessions";
// Import custom hooks
import { useTheme } from "./hooks/useTheme";
// Import utilities
import { formatSessionName } from "./utils/session";
import { API_BASE_URL } from "./lib/api-config";

function App() {
	const { isDarkMode, toggleTheme } = useTheme();
	const {
		sessions,
		activeSessionId,
		activeChatHistory,
		isInitialized,
		isSubmitting: isChatSubmitting,
		automationError,
		handleNewChat,
		handleSelectSession,
		handleDeleteSession,
		handleRenameSession,
		clearActiveChatHistory,
		handleChatSubmit: originalHandleChatSubmit,
		handleAutomateConversation,
		stopGeneration,
		// Message-level handlers
		handleEditMessage,
		handleDeleteMessage,
		handleRegenerateMessage,
		// Import/Export handlers
		handleImportConversation,
	} = useChatSessions(API_BASE_URL);

	const [availableModels, setAvailableModels] = useState([]);
	const [selectedModel, setSelectedModel] = useState("");
	const [modelsLoading, setModelsLoading] = useState(true);
	const [modelsError, setModelsError] = useState(null);
	const [isUploadingPdf, setIsUploadingPdf] = useState(false);
	const [pdfUploadStatus, setPdfUploadStatus] = useState(null);

	// Agent selector state
	const [showAgentSelector, setShowAgentSelector] = useState(false);
	const [sessionAgents, setSessionAgents] = useState({}); // Maps sessionId to agent

	// --- State for View Management (persisted to localStorage) ---
	const [currentView, setCurrentView] = useState(() => {
		const saved = localStorage.getItem("currentView");
		return saved && ["chat", "rag-hub", "guardrails"].includes(saved) ? saved : "chat";
	});
	const [ragHubView, setRagHubView] = useState(() => {
		const saved = localStorage.getItem("ragHubView");
		const validViews = ["dashboard", "raw", "pipeline", "processed", "generate-qa", "evaluation", "huggingface"];
		return saved && validViews.includes(saved) ? saved : "dashboard";
	});

	// Persist view state to localStorage
	useEffect(() => {
		localStorage.setItem("currentView", currentView);
	}, [currentView]);

	useEffect(() => {
		localStorage.setItem("ragHubView", ragHubView);
	}, [ragHubView]);

	// --- Advanced Settings State ---
	const [selectedDataset, setSelectedDataset] = useState("rag_documents");
	const [isRagEnabled, setIsRagEnabled] = useState(true);
	const [isColbertEnabled, setIsColbertEnabled] = useState(true);

	const fetchModels = useCallback(async () => {
		setModelsLoading(true);
		setModelsError(null);
		try {
			const response = await fetch(`${API_BASE_URL}/api/models`);
			if (!response.ok) {
				let errorDetail = `HTTP error! status: ${response.status}`;
				try {
					const errorData = await response.json();
					errorDetail = errorData.detail || errorDetail;
				} catch (_e) {
					/* Ignore */
				}
				throw new Error(errorDetail);
			}
			const modelsData = await response.json();
			const modelNames = modelsData.map((m) => m.name).sort();
			setAvailableModels(modelNames);

			const storedModel = localStorage.getItem("selectedModel");
			if (storedModel && modelNames.includes(storedModel)) {
				setSelectedModel(storedModel);
			} else if (
				modelNames.length > 0 &&
				(!selectedModel || !modelNames.includes(selectedModel))
			) {
				setSelectedModel(modelNames[0]);
				localStorage.setItem("selectedModel", modelNames[0]);
			} else if (modelNames.length === 0) {
				setSelectedModel("");
				localStorage.removeItem("selectedModel");
			}
		} catch (error) {
			console.error("Error fetching models:", error);
			setModelsError(`Failed to load models: ${error.message}`);
			setAvailableModels([]);
			setSelectedModel("");
			localStorage.removeItem("selectedModel");
		} finally {
			setModelsLoading(false);
		}
	}, [selectedModel]);

	useEffect(() => {
		fetchModels();
	}, [fetchModels]);

	const handleModelChange = (event) => {
		const newModel = event.target.value;
		setSelectedModel(newModel);
		if (newModel) {
			localStorage.setItem("selectedModel", newModel);
		} else {
			localStorage.removeItem("selectedModel");
		}
	};

	const handleChatSubmitWithModel = useCallback(
		async (query) => {
			if (!selectedModel) {
				console.error("No model selected. Cannot submit chat.");
				return;
			}

			// Get the selected agent for the current session (null if skipped)
			const selectedAgent = sessionAgents[activeSessionId] || null;

			// Build RAG settings from advanced settings state
			const ragSettings = {
				useRag: isRagEnabled,
				collectionName: selectedDataset !== "rag_documents" ? selectedDataset : null,
				useColbert: isColbertEnabled,
			};

			// Pass the query, model, agent, and RAG settings to the chat API
			await originalHandleChatSubmit(query, selectedModel, selectedAgent, ragSettings);
		},
		[selectedModel, originalHandleChatSubmit, sessionAgents, activeSessionId, isRagEnabled, selectedDataset, isColbertEnabled],
	);

	// --- Message Action Handlers (wrap with model and agent) ---

	const handleEditMessageWithModel = useCallback(
		async (messageId, newContent) => {
			if (!selectedModel) {
				console.error("No model selected. Cannot edit message.");
				return;
			}
			const selectedAgent = sessionAgents[activeSessionId] || null;
			await handleEditMessage(messageId, newContent, selectedModel, selectedAgent);
		},
		[selectedModel, sessionAgents, activeSessionId, handleEditMessage],
	);

	const handleRegenerateMessageWithModel = useCallback(
		async (messageId) => {
			if (!selectedModel) {
				console.error("No model selected. Cannot regenerate message.");
				return;
			}
			const selectedAgent = sessionAgents[activeSessionId] || null;
			await handleRegenerateMessage(messageId, selectedModel, selectedAgent);
		},
		[selectedModel, sessionAgents, activeSessionId, handleRegenerateMessage],
	);

	// Custom new chat handler that shows agent selector first
	const handleNewChatWithAgent = useCallback(() => {
		// Create a new chat session first
		const _newSessionId = handleNewChat();
		// Switch to chat view
		setCurrentView("chat");
		// Show agent selector for this new session
		setShowAgentSelector(true);
		// The original session will be active but won't have an agent yet
	}, [handleNewChat]);

	// Handle agent selection from the selector
	const handleAgentSelect = useCallback(
		(agentDirectory) => {
			if (activeSessionId) {
				// Store the selected agent for this session
				setSessionAgents((prev) => ({
					...prev,
					[activeSessionId]: agentDirectory,
				}));
			}
			setShowAgentSelector(false);
		},
		[activeSessionId],
	);

	// Handle agent selector cancellation
	const handleAgentCancel = useCallback(() => {
		setShowAgentSelector(false);
		// If this was a new chat without any messages, we could optionally delete it
		// For now, just hide the selector
	}, []);

	// Auto-assign null agent to existing sessions with history
	useEffect(() => {
		if (!isInitialized || !sessions) return;

		setSessionAgents((prev) => {
			const updated = { ...prev };
			let hasChanges = false;

			Object.keys(sessions).forEach((sessionId) => {
				// If session doesn't have agent info and has chat history, set to null (skip agent)
				if (
					prev[sessionId] === undefined &&
					sessions[sessionId]?.history?.length > 0
				) {
					updated[sessionId] = null; // null means "continue without agent"
					hasChanges = true;
				}
			});

			return hasChanges ? updated : prev;
		});
	}, [isInitialized, sessions]);

	const activeSessionName = useMemo(() => {
		if (
			!isInitialized ||
			!activeSessionId ||
			!sessions ||
			!sessions[activeSessionId]
		) {
			return "Chat";
		}
		return sessions[activeSessionId].name || formatSessionName(activeSessionId);
	}, [activeSessionId, sessions, isInitialized]);

	// Get the active session object for export functions
	const activeSession = useMemo(() => {
		if (!isInitialized || !activeSessionId || !sessions) {
			return null;
		}
		return sessions[activeSessionId] || null;
	}, [activeSessionId, sessions, isInitialized]);

	const handlePdfUpload = useCallback(async (file) => {
		if (!file) {
			// If called with null, just clear the status
			setPdfUploadStatus(null);
			return;
		}

		setIsUploadingPdf(true);
		setPdfUploadStatus(null);

		const formData = new FormData();
		formData.append("file", file);

		try {
			const response = await fetch(`${API_BASE_URL}/api/upload`, {
				method: "POST",
				body: formData,
			});

			const result = await response.json();

			if (!response.ok) {
				const errorMessage =
					result.detail || `HTTP error! status: ${response.status}`;
				throw new Error(errorMessage);
			}

			setPdfUploadStatus({
				success: true,
				message: result.message || "PDF uploaded successfully!",
			});
			setTimeout(() => setPdfUploadStatus(null), 5000);
		} catch (error) {
			console.error("Error uploading PDF:", error);
			setPdfUploadStatus({
				success: false,
				message: error.message || "PDF upload failed.",
			});
		} finally {
			setIsUploadingPdf(false);
		}
	}, []);

	// --- View Switching Handlers ---
	const handleViewRAGHub = useCallback(() => {
		setShowAgentSelector(false); // Close agent selector to prevent overlay blocking
		setCurrentView("rag-hub");
	}, []);

	// Handler for RAG Hub sub-view changes (from sidebar)
	const handleRagHubViewChange = useCallback((view) => {
		setCurrentView("rag-hub");
		setRagHubView(view);
	}, []);

	const handleViewGuardrails = useCallback(() => {
		setShowAgentSelector(false); // Close agent selector to prevent overlay blocking
		setCurrentView("guardrails");
	}, []);

	const handleBackToChat = useCallback(() => {
		setCurrentView("chat");
	}, []);

	// Wrapper for session selection that also switches to chat view
	const handleSelectSessionAndSwitchView = useCallback(
		(sessionId) => {
			handleSelectSession(sessionId);
			setCurrentView("chat");
		},
		[handleSelectSession],
	);

	return (
		<TooltipProvider>
			<div className="flex h-screen overflow-hidden bg-background">
				<Sidebar
					sessions={sessions}
					activeSessionId={activeSessionId}
					selectedModel={selectedModel}
					onNewChat={handleNewChatWithAgent}
					onSelectSession={handleSelectSessionAndSwitchView}
					onDeleteSession={handleDeleteSession}
					onRenameSession={handleRenameSession}
					onAutomateConversation={handleAutomateConversation}
					isSubmitting={isChatSubmitting}
					automationError={automationError}
					isInitialized={isInitialized}
					// PDF Upload props
					onUploadPdf={handlePdfUpload}
					isUploadingPdf={isUploadingPdf}
					pdfUploadStatus={pdfUploadStatus}
					// View switching props
					onViewRAGHub={handleViewRAGHub}
					onViewGuardrails={handleViewGuardrails}
					// Theme props
					isDarkMode={isDarkMode}
					toggleTheme={toggleTheme}
					// RAG Hub navigation props
					currentView={currentView}
					ragHubView={ragHubView}
					onRagHubViewChange={handleRagHubViewChange}
				/>
				{/* Main content area */}
				<div className="flex-1 flex flex-col overflow-hidden">
					{currentView === "chat" ? (
						<ChatInterface
							key={`${activeSessionId || "no-session"}-${activeChatHistory.length}`}
							activeSessionId={activeSessionId}
							activeSessionName={activeSessionName}
							activeSession={activeSession}
							chatHistory={activeChatHistory}
							onSubmit={handleChatSubmitWithModel}
							onClearHistory={clearActiveChatHistory}
							onImportConversation={handleImportConversation}
							isSubmitting={isChatSubmitting}
							onStopGeneration={stopGeneration}
							availableModels={availableModels}
							selectedModel={selectedModel}
							onModelChange={handleModelChange}
							modelsLoading={modelsLoading}
							modelsError={modelsError}
							isInitialized={isInitialized}
							// Pass agent selector info
							showAgentSelector={showAgentSelector}
							sessionAgents={sessionAgents}
							onRefreshModels={fetchModels}
							// Message action handlers
							onEditMessage={handleEditMessageWithModel}
							onDeleteMessage={handleDeleteMessage}
							onRegenerateMessage={handleRegenerateMessageWithModel}
							// Advanced Settings props
							selectedDataset={selectedDataset}
							onDatasetChange={setSelectedDataset}
							isRagEnabled={isRagEnabled}
							onRagEnabledChange={setIsRagEnabled}
							isColbertEnabled={isColbertEnabled}
							onColbertEnabledChange={setIsColbertEnabled}
						/>
					) : currentView === "rag-hub" ? (
						<RAGBenchmarkHub
							onBack={handleBackToChat}
							isDarkMode={isDarkMode}
							currentView={ragHubView}
							onViewChange={setRagHubView}
						/>
					) : (
						<GuardrailsEditor onBack={handleBackToChat} isDarkMode={isDarkMode} />
					)}
				</div>

				{/* Render AgentSelector as overlay when needed */}
				{showAgentSelector && (
					<AgentSelector
						onAgentSelect={handleAgentSelect}
						onCancel={handleAgentCancel}
					/>
				)}
			</div>
		</TooltipProvider>
	);
}

export default App;
