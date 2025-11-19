// HIA/frontend/src/hooks/usePersistentSessions.js
import { useCallback, useEffect, useRef, useState } from "react";

// Helper function to generate a new session ID
const generateNewSessionId = (counter) => `new-chat-${counter}`;

// Helper function to format session names (used as fallback)
const _formatSessionIdFallback = (sessionId) => {
	if (!sessionId) return "Chat";
	return sessionId.replace(/-/g, " ").replace(/^./, (str) => str.toUpperCase());
};

/**
 * Manages the sessions data, persistence, and basic CRUD operations.
 * @param {string | null} initialActiveId - The active session ID potentially loaded by useLocalStorage.
 * @param {(id: string | null) => void} setActiveSessionIdDirectly - The setter from useLocalStorage for activeSessionId.
 * @returns {{
 * sessions: Record<string, { name: string | null, history: Array<any> }>,
 * setSessions: React.Dispatch<React.SetStateAction<Record<string, { name: string | null, history: Array<any> }>>>,
 * isInitialized: boolean,
 * addSession: () => string,
 * deleteSession: (sessionIdToDelete: string, currentActiveId: string | null) => string | null,
 * renameSession: (sessionId: string, newName: string) => void,
 * clearSessionHistory: (sessionId: string) => void
 * }}
 */
export function usePersistentSessions(
	initialActiveId,
	setActiveSessionIdDirectly,
) {
	const [sessions, setSessions] = useState({});
	const [isInitialized, setIsInitialized] = useState(false);
	const sessionCounterRef = useRef(1);
	const setActiveSessionIdRef = useRef(setActiveSessionIdDirectly);
	
	// Update ref when the function changes
	useEffect(() => {
		setActiveSessionIdRef.current = setActiveSessionIdDirectly;
	}, [setActiveSessionIdDirectly]);

	// --- Effect 1: Load from localStorage and Initialize ONCE on mount ---
	useEffect(() => {
		console.log("[Init] Attempting to load sessions from localStorage...");
		let loadedSessions = {};
		let nextSessionCounter = 1;
		let activeIdToSet = initialActiveId; // Start with the value from useLocalStorage

		try {
			const storedSessions = window.localStorage.getItem("chatSessions");
			if (storedSessions) {
				const parsedSessions = JSON.parse(storedSessions);
				// Validate structure
				if (
					typeof parsedSessions === "object" &&
					parsedSessions !== null &&
					Object.values(parsedSessions).every(
						(s) =>
							s &&
							typeof s === "object" &&
							Object.hasOwn(s, "history") &&
							Array.isArray(s.history),
					)
				) {
					let maxNum = 0;
					loadedSessions = Object.entries(parsedSessions).reduce(
						(acc, [id, sessionData]) => {
							acc[id] = {
								name: sessionData.name !== undefined ? sessionData.name : null,
								history: sessionData.history,
							};
							if (id.startsWith("new-chat-")) {
								const num = parseInt(id.replace("new-chat-", ""), 10);
								if (!Number.isNaN(num) && num > maxNum) maxNum = num;
							}
							return acc;
						},
						{},
					);
					nextSessionCounter = maxNum + 1;
					console.log("[Init] Successfully loaded sessions:", loadedSessions);
				} else {
					console.warn("[Init] Invalid data format in localStorage. Clearing.");
					window.localStorage.removeItem("chatSessions");
					window.localStorage.removeItem("activeSessionId");
					activeIdToSet = null; // Reset active ID
				}
			} else {
				console.log("[Init] No sessions found in localStorage.");
				activeIdToSet = null; // Reset active ID if storage is empty
			}
		} catch (error) {
			console.error(
				"[Init] Error reading/parsing sessions from localStorage:",
				error,
			);
			loadedSessions = {};
			window.localStorage.removeItem("chatSessions");
			window.localStorage.removeItem("activeSessionId");
			activeIdToSet = null;
		}

		sessionCounterRef.current = nextSessionCounter;

		// --- Ensure a session exists and is active ---
		const loadedSessionIds = Object.keys(loadedSessions);

		if (loadedSessionIds.length === 0) {
			console.log("[Init] No valid sessions loaded. Creating initial session.");
			const firstSessionId = generateNewSessionId(sessionCounterRef.current);
			sessionCounterRef.current++;
			loadedSessions[firstSessionId] = { name: null, history: [] };
			activeIdToSet = firstSessionId; // Set the new one as active
		} else {
			// Validate the potentially loaded activeIdToSet
			if (!activeIdToSet || !loadedSessions[activeIdToSet]) {
				console.log(
					`[Init] Stored activeSessionId ('${activeIdToSet}') is invalid. Selecting first available.`,
				);
				activeIdToSet = loadedSessionIds[0]; // Select the first loaded session
			}
		}

		setSessions(loadedSessions); // Update sessions state
		// Crucially, update the activeSessionId state via the passed setter *after* potential correction
		if (activeIdToSet !== initialActiveId) {
			setActiveSessionIdRef.current(activeIdToSet);
		}
		setIsInitialized(true); // Mark initialization complete
		console.log(
			"[Init] Initialization complete. Active session:",
			activeIdToSet,
		);

		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []); // Only run once on mount - don't re-initialize on activeSessionId changes

	// --- Effect 2: Save sessions to localStorage whenever they change (AFTER init) ---
	useEffect(() => {
		if (isInitialized && sessions) {
			// Check sessions directly
			if (Object.keys(sessions).length > 0) {
				const totalMessages = Object.values(sessions).reduce(
					(sum, session) => sum + (session.history?.length || 0),
					0,
				);
				console.log(
					`[Save] Saving ${Object.keys(sessions).length} session(s) with ${totalMessages} total message(s) to localStorage`,
				);
				try {
					window.localStorage.setItem("chatSessions", JSON.stringify(sessions));
					console.log("[Save] Successfully saved to localStorage");
				} catch (error) {
					console.error("[Save] Error saving sessions to localStorage:", error);
				}
			} else {
				// If initialized and sessions become empty, clear storage
				console.log("[Save] Sessions are empty, clearing localStorage.");
				window.localStorage.removeItem("chatSessions");
				// Note: activeSessionId is cleared separately by its own hook if needed
			}
		}
	}, [sessions, isInitialized]); // Run when sessions change or initialization completes

	// --- Session Management Functions ---

	const addSession = useCallback(() => {
		const newSessionId = generateNewSessionId(sessionCounterRef.current);
		sessionCounterRef.current++;
		setSessions((prevSessions) => ({
			...prevSessions,
			[newSessionId]: { name: null, history: [] },
		}));
		return newSessionId; // Return the new ID so the caller can set it active
	}, []); // No dependencies needed here

	const deleteSession = useCallback((sessionIdToDelete, currentActiveId) => {
		let nextActiveId = currentActiveId; // Assume active ID doesn't change initially
		let deleted = false;

		setSessions((prevSessions) => {
			if (!prevSessions || !prevSessions[sessionIdToDelete])
				return prevSessions; // Already deleted or invalid

			const currentIds = Object.keys(prevSessions);
			if (currentIds.length <= 1) {
				console.warn("Cannot delete the last session (from hook)."); // Log warning
				return prevSessions; // Don't modify state
			}

			const updatedSessions = { ...prevSessions };
			delete updatedSessions[sessionIdToDelete];
			deleted = true; // Mark as deleted

			// Determine the next active ID *if* the deleted one was active
			if (currentActiveId === sessionIdToDelete) {
				const remainingIds = Object.keys(updatedSessions);
				nextActiveId = remainingIds.length > 0 ? remainingIds[0] : null;
			}
			return updatedSessions;
		});

		// Return the ID that should become active *after* deletion
		// Return null if the deletion didn't happen (e.g., last session)
		return deleted ? nextActiveId : currentActiveId;
	}, []); // No state dependencies needed here

	const renameSession = useCallback((sessionId, newName) => {
		const trimmedName = newName.trim();
		if (!trimmedName) {
			console.warn("Attempted to rename session with empty name.");
			return; // Don't allow empty names
		}
		setSessions((prevSessions) => {
			if (!prevSessions[sessionId]) return prevSessions; // Check if session exists
			return {
				...prevSessions,
				[sessionId]: {
					...prevSessions[sessionId],
					name: trimmedName,
				},
			};
		});
	}, []); // No state dependencies needed here

	const clearSessionHistory = useCallback((sessionId) => {
		setSessions((prevSessions) => {
			if (!prevSessions[sessionId]) return prevSessions; // Check if session exists
			// Confirm before clearing (moved confirmation to caller if needed)
			// const sessionName = prevSessions[sessionId].name || formatSessionIdFallback(sessionId);
			// if (window.confirm(`Are you sure you want to clear the history for "${sessionName}"?`)) {
			return {
				...prevSessions,
				[sessionId]: {
					...prevSessions[sessionId],
					history: [],
				},
			};
			// }
			// return prevSessions; // Return unchanged if confirmation is cancelled
		});
	}, []); // No state dependencies needed here

	return {
		sessions,
		setSessions, // Expose setter for useChatApi
		isInitialized,
		addSession,
		deleteSession,
		renameSession,
		clearSessionHistory,
	};
}
