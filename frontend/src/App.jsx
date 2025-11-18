// HIA/frontend/src/App.jsx
import { useCallback, useEffect, useMemo, useState } from "react";
import "./index.css";

import { AgentSelector } from "./components/AgentSelector";
import { ChatInterface } from "./components/ChatInterface";
import { DocumentViewer } from "./components/DocumentViewer"; // Import the new component

// Import components
import { Sidebar } from "./components/Sidebar";
import { TooltipProvider } from "./components/ui/tooltip";
import { useChatSessions } from "./hooks/useChatSessions";
// Import custom hooks
import { useTheme } from "./hooks/useTheme";

// Define the backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

	// --- State for View Management ---
	const [currentView, setCurrentView] = useState("chat"); // 'chat' or 'documents'

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

			// Pass the query, model, and agent to the chat API
			await originalHandleChatSubmit(query, selectedModel, selectedAgent);
		},
		[selectedModel, originalHandleChatSubmit, sessionAgents, activeSessionId],
	);

	// Custom new chat handler that shows agent selector first
	const handleNewChatWithAgent = useCallback(() => {
		// Create a new chat session first
		const _newSessionId = handleNewChat();
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

	const formatSessionIdFallback = useCallback((sessionId) => {
		if (!sessionId) return "Chat";
		return sessionId
			.replace(/-/g, " ")
			.replace(/^./, (str) => str.toUpperCase());
	}, []);

	const activeSessionName = useMemo(() => {
		if (
			!isInitialized ||
			!activeSessionId ||
			!sessions ||
			!sessions[activeSessionId]
		) {
			return "Chat";
		}
		return (
			sessions[activeSessionId].name || formatSessionIdFallback(activeSessionId)
		);
	}, [activeSessionId, sessions, isInitialized, formatSessionIdFallback]);

	const downloadActiveChatHistory = useCallback(() => {
		const currentSession =
			sessions && activeSessionId ? sessions[activeSessionId] : null;
		if (
			!currentSession ||
			!Array.isArray(currentSession.history) ||
			currentSession.history.length === 0
		) {
			alert("No history to download for the current chat.");
			return;
		}
		const historyToDownload = currentSession.history;
		const json = JSON.stringify(historyToDownload, null, 2);
		const blob = new Blob([json], { type: "application/json" });
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		const downloadName = currentSession.name
			? currentSession.name.replace(/[^a-z0-9]/gi, "_").toLowerCase()
			: activeSessionId;
		link.download = `${downloadName}_history.json`;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
		URL.revokeObjectURL(url);
	}, [activeSessionId, sessions]);

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
	const handleViewDocuments = useCallback(() => {
		console.log("Switching to documents view");
		setShowAgentSelector(false); // Close agent selector to prevent overlay blocking
		setCurrentView("documents");
	}, []);

	const handleBackToChat = useCallback(() => {
		console.log("Back to Chat button clicked - switching to chat view");
		setShowAgentSelector(false); // Close agent selector for consistency
		setCurrentView("chat");
	}, []);

	return (
		<TooltipProvider>
			<div className="flex h-screen overflow-hidden bg-background">
				<Sidebar
					sessions={sessions}
					activeSessionId={activeSessionId}
					selectedModel={selectedModel}
					onNewChat={handleNewChatWithAgent}
					onSelectSession={handleSelectSession}
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
					// View switching prop
					onViewDocuments={handleViewDocuments}
				/>
				{/* Main content area */}
				<div className="flex-1 flex flex-col overflow-hidden">
					{currentView === "chat" ? (
						<ChatInterface
							key={`${activeSessionId || "no-session"}-${activeChatHistory.length}`}
							activeSessionId={activeSessionId}
							activeSessionName={activeSessionName}
							chatHistory={activeChatHistory}
							onSubmit={handleChatSubmitWithModel}
							onClearHistory={clearActiveChatHistory}
							onDownloadHistory={downloadActiveChatHistory}
							isSubmitting={isChatSubmitting}
							isDarkMode={isDarkMode}
							toggleTheme={toggleTheme}
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
						/>
					) : (
						<DocumentViewer onBackToChat={handleBackToChat} />
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
