// HIA/frontend/src/hooks/useChatSessions.js
import { useCallback, useMemo } from "react";
import { useActiveSession } from "./useActiveSession";
import { useChatApi } from "./useChatApi";
import { usePersistentSessions } from "./usePersistentSessions";

// Helper function (can be moved to utils)
const formatSessionIdFallback = (sessionId) => {
	if (!sessionId) return "Chat";
	return sessionId.replace(/-/g, " ").replace(/^./, (str) => str.toUpperCase());
};

/**
 * Orchestrator hook that combines session management, active session, and API logic.
 * @param {string} apiBaseUrl - The base URL for the backend API.
 */
export function useChatSessions(apiBaseUrl) {
	// --- Core Hooks ---
	const { activeSessionId, setActiveSessionId } = useActiveSession();
	const {
		sessions,
		setSessions, // Pass this setter to the API hook
		isInitialized,
		addSession,
		deleteSession: deleteSessionInternal, // Rename internal function
		renameSession,
		clearSessionHistory: clearSessionHistoryInternal, // Rename internal function
	} = usePersistentSessions(activeSessionId, setActiveSessionId); // Pass initial activeId and direct setter

	const {
		isSubmitting,
		automationError,
		handleChatSubmit: handleChatSubmitRaw,
		handleAutomateConversation,
		setAutomationError, // Get setter from API hook
		stopGeneration,
	} = useChatApi(apiBaseUrl, activeSessionId, setSessions); // Provide dependencies

	// --- Combined Session Management Handlers ---

	// Handle creating a new chat and making it active
	const handleNewChat = useCallback(() => {
		const newSessionId = addSession();
		setActiveSessionId(newSessionId);
		setAutomationError(null); // Clear errors on new chat
	}, [addSession, setActiveSessionId, setAutomationError]);

	// Handle selecting a session (clears automation error)
	const handleSelectSession = useCallback(
		(sessionId) => {
			// Basic check if ID exists before setting active
			if (sessions?.[sessionId]) {
				setActiveSessionId(sessionId);
				setAutomationError(null);
			} else {
				console.warn(
					`useChatSessions: Attempted to select non-existent session ${sessionId}`,
				);
				// Optionally select the first available if the target is invalid
				const firstId = Object.keys(sessions)[0];
				if (firstId) setActiveSessionId(firstId);
			}
		},
		[sessions, setActiveSessionId, setAutomationError],
	);

	// Handle deleting a session and updating the active ID if necessary
	const handleDeleteSession = useCallback(
		(sessionIdToDelete) => {
			// Confirmation dialog
			const sessionName =
				sessions[sessionIdToDelete]?.name ||
				formatSessionIdFallback(sessionIdToDelete);
			if (
				!window.confirm(`Are you sure you want to delete "${sessionName}"?`)
			) {
				return;
			}

			// Call the internal delete function which returns the next active ID
			const nextActiveId = deleteSessionInternal(
				sessionIdToDelete,
				activeSessionId,
			);

			// If the active ID changed as a result of deletion, update it
			if (nextActiveId !== activeSessionId) {
				setActiveSessionId(nextActiveId);
				setAutomationError(null); // Clear errors if active session changed
			}
		},
		[
			sessions,
			deleteSessionInternal,
			activeSessionId,
			setActiveSessionId,
			setAutomationError,
		],
	);

	// Handle clearing history (with confirmation)
	const clearActiveChatHistory = useCallback(() => {
		if (!activeSessionId || !sessions[activeSessionId]) return;
		const sessionName =
			sessions[activeSessionId].name ||
			formatSessionIdFallback(activeSessionId);
		if (
			window.confirm(
				`Are you sure you want to clear the history for "${sessionName}"?`,
			)
		) {
			clearSessionHistoryInternal(activeSessionId);
			setAutomationError(null); // Clear errors
		}
	}, [
		activeSessionId,
		sessions,
		clearSessionHistoryInternal,
		setAutomationError,
	]);

	// --- Derived State ---

	// Derive active chat history only when initialized and session exists
	const activeChatHistory = useMemo(() => {
		if (
			!isInitialized ||
			!sessions ||
			!activeSessionId ||
			!sessions[activeSessionId]
		) {
			return [];
		}
		const history = sessions[activeSessionId].history;
		return Array.isArray(history) ? history : [];
	}, [activeSessionId, sessions, isInitialized]);

	// --- Wrap handleChatSubmit to include conversation history ---
	const handleChatSubmit = useCallback(
		async (query, model, agent = null) => {
			// Get current conversation history and pass it to the API handler
			const currentHistory = activeChatHistory || [];
			await handleChatSubmitRaw(query, model, agent, currentHistory);
		},
		[handleChatSubmitRaw, activeChatHistory],
	);

	// --- Exposed API ---
	return {
		sessions, // The full sessions object { id: { name, history } }
		activeSessionId, // The current active session ID
		activeChatHistory, // The history array for the active session
		isInitialized, // Flag indicating if loading/initialization is complete
		isSubmitting, // Flag indicating if an API call is in progress
		automationError, // Error message from the automation process

		// Combined/wrapped handlers
		handleNewChat,
		handleSelectSession,
		handleDeleteSession,
		clearActiveChatHistory,

		// Direct handlers from sub-hooks
		handleRenameSession: renameSession, // Pass rename directly
		handleChatSubmit, // Interactive chat submit from API hook
		handleAutomateConversation, // Automated chat submit from API hook
		stopGeneration, // Stop token generation

		// Potentially useful utilities (optional)
		// downloadActiveChatHistory, // This could be rebuilt here or moved to its own hook/util
	};
}
