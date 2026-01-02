// HIA/frontend/src/hooks/useChatSessions.js
import { useCallback, useMemo } from "react";
import { formatSessionName } from "../utils/session";
import { useActiveSession } from "./useActiveSession";
import { useChatApi } from "./useChatApi";
import { usePersistentSessions } from "./usePersistentSessions";

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
		// Message-level operations
		deleteMessage,
		updateMessage,
		getMessagesUntil,
		truncateHistoryAt,
		// Import operations
		importAsNewSession,
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
				formatSessionName(sessionIdToDelete);
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
			formatSessionName(activeSessionId);
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

	// --- Message-level Handlers ---

	/**
	 * Handle editing a user message - updates the message and triggers regeneration
	 * This removes the message and everything after it, then submits the new message
	 */
	const handleEditMessage = useCallback(
		async (messageId, newContent, model, agent = null) => {
			if (!activeSessionId) return;

			// Get history up to the message being edited
			const historyBefore = getMessagesUntil(activeSessionId, messageId);

			// Truncate history at the edited message (removes it and everything after)
			truncateHistoryAt(activeSessionId, messageId);

			// Submit the new message (this will add it to history and get a response)
			await handleChatSubmitRaw(newContent, model, agent, historyBefore);
		},
		[activeSessionId, getMessagesUntil, truncateHistoryAt, handleChatSubmitRaw],
	);

	/**
	 * Handle deleting a message from the active session
	 */
	const handleDeleteMessage = useCallback(
		(messageId) => {
			if (!activeSessionId) return;
			deleteMessage(activeSessionId, messageId);
		},
		[activeSessionId, deleteMessage],
	);

	/**
	 * Handle regenerating an assistant message
	 * Removes the assistant message and re-submits the previous user message
	 */
	const handleRegenerateMessage = useCallback(
		async (messageId, model, agent = null) => {
			if (!activeSessionId) return;

			// Get history up to the message being regenerated
			const historyBefore = getMessagesUntil(activeSessionId, messageId);

			// Find the last user message before this one
			const lastUserMessage = [...historyBefore]
				.reverse()
				.find((msg) => msg.sender === "user");

			if (!lastUserMessage) {
				console.warn("Cannot regenerate: no previous user message found");
				return;
			}

			// Truncate history at the assistant message (removes it)
			truncateHistoryAt(activeSessionId, messageId);

			// Get history without the user message we're about to re-submit
			const historyWithoutLastUser = historyBefore.filter(
				(msg) => msg.id !== lastUserMessage.id
			);

			// Re-submit the user message to get a new response
			await handleChatSubmitRaw(
				lastUserMessage.text,
				model,
				agent,
				historyWithoutLastUser
			);
		},
		[activeSessionId, getMessagesUntil, truncateHistoryAt, handleChatSubmitRaw],
	);

	/**
	 * Handle importing a conversation as a new session
	 * Creates the session and makes it active
	 */
	const handleImportConversation = useCallback(
		(importedData) => {
			const newSessionId = importAsNewSession(importedData);
			setActiveSessionId(newSessionId);
			setAutomationError(null);
			return newSessionId;
		},
		[importAsNewSession, setActiveSessionId, setAutomationError],
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

		// Message-level handlers
		handleEditMessage, // Edit a user message and regenerate response
		handleDeleteMessage, // Delete a specific message
		handleRegenerateMessage, // Regenerate an assistant response

		// Import/Export handlers
		handleImportConversation, // Import a conversation as a new session
	};
}
