/**
 * Session-related utility functions.
 * Centralizes session ID formatting to eliminate duplication.
 */

/**
 * Format a session ID into a human-readable name.
 *
 * @param {string|null|undefined} sessionId - The session ID to format
 * @returns {string} A formatted session name
 *
 * @example
 * formatSessionName("my-chat-session") // "My chat session"
 * formatSessionName(null) // "Chat"
 * formatSessionName("") // "Chat"
 */
export function formatSessionName(sessionId) {
	if (!sessionId) return "Chat";

	return sessionId
		.replace(/-/g, " ")
		.replace(/^./, (str) => str.toUpperCase());
}

/**
 * Generate a unique session ID.
 *
 * @returns {string} A unique session ID based on timestamp
 *
 * @example
 * generateSessionId() // "chat-1703520000000"
 */
export function generateSessionId() {
	return `chat-${Date.now()}`;
}

/**
 * Validate that a session ID exists in the sessions object.
 *
 * @param {string} sessionId - The session ID to validate
 * @param {Object} sessions - The sessions object to check against
 * @returns {boolean} True if the session exists
 */
export function isValidSession(sessionId, sessions) {
	return sessionId && sessions && sessions[sessionId] !== undefined;
}
