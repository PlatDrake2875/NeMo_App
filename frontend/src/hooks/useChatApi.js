// HIA/frontend/src/hooks/useChatApi.js
import { useCallback, useRef, useState } from "react";

// Assuming API_BASE_URL is defined elsewhere or passed in correctly
// const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const CHAT_API_URL_SUFFIX = "/api/chat";
const AUTOMATE_API_URL_SUFFIX = "/api/automate_conversation";

/**
 * Handles API interactions for chat and automation.
 * @param {string} apiBaseUrl - The base URL for the API.
 * @param {string | null} activeSessionId - The currently active session ID.
 * @param {React.Dispatch<React.SetStateAction<Record<string, { name: string | null, history: Array<any>, model?: string }>>>} setSessions - State setter for the sessions object.
 * @returns {{
 * isSubmitting: boolean,
 * automationError: string | null,
 * handleChatSubmit: (query: string, model: string) => Promise<void>,
 * handleAutomateConversation: (jsonInputString: string, model: string, automationTask?: string | null) => Promise<void>, // Added automationTask
 * setAutomationError: React.Dispatch<React.SetStateAction<string | null>>
 * }}
 */
export function useChatApi(apiBaseUrl, activeSessionId, setSessions) {
	const chatApiUrl = `${apiBaseUrl}${CHAT_API_URL_SUFFIX}`;
	const automateApiUrl = `${apiBaseUrl}${AUTOMATE_API_URL_SUFFIX}`;

	const [isSubmitting, setIsSubmitting] = useState(false);
	const [automationError, setAutomationError] = useState(null);

	// AbortController to cancel ongoing requests
	const abortControllerRef = useRef(null);

	// Function to stop ongoing generation
	const stopGeneration = useCallback(() => {
		if (abortControllerRef.current) {
			console.log("Aborting current request...");
			abortControllerRef.current.abort();
			abortControllerRef.current = null;
		}
	}, []);

	// --- Interactive Chat Submission ---
	const handleChatSubmit = useCallback(
		async (query, model, agent = null, conversationHistory = [], ragSettings = {}) => {
			if (!activeSessionId || !model || isSubmitting) {
				console.warn("API Hook: Chat submission prevented.", {
					activeSessionId,
					model,
					isSubmitting,
				});
				return;
			}

			// Extract RAG settings with defaults
			const {
				useRag = true,
				collectionName = null,
				useColbert = true,
			} = ragSettings;
			const reqId = `req_${Date.now()}`; // Simple request ID
			console.log(
				`API Hook: [${activeSessionId} - ${reqId}] SUBMIT starting for model: ${model}`,
			);

			// Create new AbortController for this request
			abortControllerRef.current = new AbortController();

			setIsSubmitting(true);
			setAutomationError(null);

			// Format conversation history for backend (strip id field, keep only sender and text)
			const formattedHistory = conversationHistory.map((msg) => ({
				sender: msg.sender,
				text: msg.text,
			}));

			console.log(
				`API Hook: [${activeSessionId} - ${reqId}] Sending ${formattedHistory.length} previous messages as context`,
			);

			const userMessage = {
				sender: "user",
				text: query,
				id: `user-${Date.now()}`,
			};
			const botMessageId = `bot-${Date.now()}-${Math.random()}`;
			const botMessagePlaceholder = {
				id: botMessageId,
				sender: "bot",
				text: "...",
				isLoading: true,
			};

			setSessions((prevSessions) => {
				if (!prevSessions[activeSessionId]) return prevSessions;
				const currentHistory = prevSessions[activeSessionId].history || [];
				const newHistory = [
					...currentHistory,
					userMessage,
					botMessagePlaceholder,
				];
				return {
					...prevSessions,
					[activeSessionId]: {
						...prevSessions[activeSessionId],
						history: newHistory,
					},
				};
			});

			let accumulatedBotResponse = "";
			let streamError = null;
			let wasAborted = false;

			try {
				const response = await fetch(chatApiUrl, {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						Accept: "text/event-stream",
					},
					body: JSON.stringify({
						query,
						model,
						history: formattedHistory, // Include conversation history for context
						...(agent && { agent_name: agent }), // Include agent_name only if agent is provided
						use_rag: useRag,
						...(collectionName && { collection_name: collectionName }),
						use_colbert: useColbert,
					}),
					signal: abortControllerRef.current.signal,
				});

				if (!response.ok || !response.body) {
					let errorDetail = `HTTP error! status: ${response.status}`;
					try {
						const errorData = await response.text();
						errorDetail = `${errorDetail}: ${errorData.substring(0, 150)}`;
					} catch (_e) {}
					throw new Error(errorDetail);
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let buffer = "";

				// eslint-disable-next-line no-constant-condition
				while (true) {
					const { value, done: readerDone } = await reader.read();
					if (readerDone) {
						console.log(
							`API Hook: [${activeSessionId} - ${reqId}] Stream finished.`,
						);
						break;
					}
					buffer += decoder.decode(value, { stream: true });
					let eolIndex;
					while ((eolIndex = buffer.indexOf("\n\n")) >= 0) {
						const message = buffer.substring(0, eolIndex).trim();
						buffer = buffer.substring(eolIndex + 2);
						if (message.startsWith("data:")) {
							const jsonString = message.substring(5).trim();
							if (jsonString) {
								// Check for [DONE] signal first
								if (jsonString === "[DONE]") {
									console.log(
										`API Hook: [${activeSessionId} - ${reqId}] Received [DONE] signal via SSE.`,
									);
									// Don't try to parse [DONE] as JSON, just continue
								} else {
									try {
										const parsedData = JSON.parse(jsonString);
										let contentToAdd = "";

										// Handle different response formats
										if (parsedData.token) {
											// Old format: direct token
											contentToAdd = parsedData.token;
										} else if (parsedData.choices?.[0]?.delta?.content) {
											// OpenAI ChatCompletion format
											contentToAdd = parsedData.choices[0].delta.content;
										} else if (
											parsedData.status === "done" ||
											parsedData.done
										) {
											// Handle 'done' from backend
											console.log(
												`API Hook: [${activeSessionId} - ${reqId}] Received 'done' status via SSE.`,
											);
										} else if (parsedData.error) {
											console.error(
												`API Hook: [${activeSessionId} - ${reqId}] Error received via SSE:`,
												parsedData.error,
											);
											throw new Error(`Stream error: ${parsedData.error}`);
										}

										// If we have content to add, update the UI
										if (contentToAdd) {
											accumulatedBotResponse += contentToAdd;
											setSessions((prevSessions) => {
												if (!prevSessions[activeSessionId]) return prevSessions;
												const history =
													prevSessions[activeSessionId].history || [];
												const botIndex = history.findIndex(
													(msg) => msg.id === botMessageId,
												);
												if (botIndex === -1) return prevSessions;
												const updatedMsg = {
													...history[botIndex],
													text: accumulatedBotResponse,
													isLoading: true,
												};
												const newHistory = [
													...history.slice(0, botIndex),
													updatedMsg,
													...history.slice(botIndex + 1),
												];
												return {
													...prevSessions,
													[activeSessionId]: {
														...prevSessions[activeSessionId],
														history: newHistory,
													},
												};
											});
										}
									} catch (e) {
										console.error(
											`API Hook: [${activeSessionId} - ${reqId}] Error parsing JSON from SSE line:`,
											jsonString,
											e,
										);
									}
								}
							}
						} else if (message) {
							console.log(
								`API Hook: [${activeSessionId} - ${reqId}] Received non-data SSE line: ${message}`,
							);
						}
					}
				}
			} catch (error) {
				// Check if the request was aborted by the user
				if (error.name === "AbortError") {
					console.log(
						`API Hook: [${activeSessionId} - ${reqId}] Request aborted by user`,
					);
					console.log(
						`API Hook: [${activeSessionId} - ${reqId}] Preserving ${accumulatedBotResponse.length} characters of partial response`,
					);
					wasAborted = true;
					// Don't set streamError - we want to keep the accumulated response
				} else {
					console.error(
						`API Hook: [${activeSessionId} - ${reqId}] Chat stream error:`,
						error,
					);
					streamError = error;
				}
			} finally {
				setSessions((prevSessions) => {
					if (!prevSessions[activeSessionId]) return prevSessions;
					const history = prevSessions[activeSessionId].history || [];
					const botIndex = history.findIndex((msg) => msg.id === botMessageId);
					if (botIndex === -1) return prevSessions;

					// Determine the final message text
					let finalText;
					if (streamError) {
						finalText = `⚠️ ${streamError?.message || "Unknown stream error"}`;
					} else if (accumulatedBotResponse) {
						finalText = accumulatedBotResponse;
						if (wasAborted) {
							console.log(
								`API Hook: [${activeSessionId} - ${reqId}] Saving partial response to conversation history (${finalText.length} chars)`,
							);
						}
					} else {
						finalText = "...";
					}

					const finalMsg = {
						...history[botIndex],
						text: finalText,
						isLoading: false,
					};
					const newHistory = [
						...history.slice(0, botIndex),
						finalMsg,
						...history.slice(botIndex + 1),
					];
					return {
						...prevSessions,
						[activeSessionId]: {
							...prevSessions[activeSessionId],
							history: newHistory,
						},
					};
				});

				// Clean up the AbortController
				abortControllerRef.current = null;

				setIsSubmitting(false);

				if (wasAborted) {
					console.log(
						`API Hook: [${activeSessionId} - ${reqId}] SUBMIT finished (stopped by user, partial response saved to history).`,
					);
				} else {
					console.log(
						`API Hook: [${activeSessionId} - ${reqId}] SUBMIT finished.`,
					);
				}
			}
		},
		[activeSessionId, isSubmitting, setSessions, chatApiUrl],
	);

	// --- Automated Conversation Submission ---
	const handleAutomateConversation = useCallback(
		async (jsonInputString, model, automationTask = null) => {
			if (!activeSessionId || !model || isSubmitting) {
				setAutomationError(
					"Automation cannot start: Another process is running, or no session/model selected.",
				);
				console.warn("API Hook: Automate submission prevented.", {
					activeSessionId,
					model,
					isSubmitting,
				});
				return;
			}

			let parsedJsonInputs;
			try {
				const jsonData = JSON.parse(jsonInputString);
				// Assuming the JSON has an "inputs" array of strings, as per your Sidebar's textarea placeholder
				if (
					!jsonData ||
					!Array.isArray(jsonData.inputs) ||
					!jsonData.inputs.every((i) => typeof i === "string")
				) {
					throw new Error(
						'Invalid JSON format. Expected: { "inputs": ["message1", "message2", ...] }',
					);
				}
				parsedJsonInputs = jsonData.inputs;
				if (parsedJsonInputs.length === 0) {
					throw new Error('JSON "inputs" array cannot be empty.');
				}
			} catch (error) {
				setAutomationError(
					`Automation failed: Invalid JSON input. ${error.message}`,
				);
				return;
			}

			// Transform the array of input strings into the conversation_history format
			// Assigning 'user' role by default. If roles are needed, JSON structure should be richer.
			const formattedConversationHistory = parsedJsonInputs.map((content) => ({
				role: "user", // Defaulting to 'user' for inputs from the JSON array
				content: content,
			}));

			console.log(
				`API Hook: [${activeSessionId}] AUTOMATE starting with model: ${model}, task: ${automationTask || "N/A"}`,
			);
			console.log(
				`API Hook: [${activeSessionId}] Sending conversation history:`,
				formattedConversationHistory,
			);
			setIsSubmitting(true);
			setAutomationError(null);

			// Optionally, clear current history or add a placeholder
			// For now, we'll append the automation result as a system message.
			// If you want to replace history, you can do:
			// setSessions(prevSessions => ({ ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: [] } }));

			try {
				const response = await fetch(automateApiUrl, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						conversation_history: formattedConversationHistory, // Corrected field name and format
						model: model,
						automation_task: automationTask, // Pass the task
						// config_params: {} // Add if you have specific config params
					}),
				});

				if (!response.ok) {
					let errorDetail = `HTTP error! status: ${response.status}`;
					try {
						const errorData = await response.json(); // FastAPI 422 errors are JSON
						errorDetail = `${errorDetail}: ${JSON.stringify(errorData.detail || errorData)}`;
					} catch (_e) {
						try {
							const textError = await response.text();
							errorDetail = `${errorDetail}: ${textError.substring(0, 200)}`;
						} catch (_e2) {}
					}
					throw new Error(errorDetail);
				}

				const result = await response.json(); // Expects AutomateResponse: {status, message, data}
				console.log(
					`API Hook: [${activeSessionId}] AUTOMATE success response:`,
					result,
				);

				if (result.status === "success" && result.data) {
					let automationMessageContent = `Automation Result: ${result.message || "Task processed."}`;
					if (result.data.summary) {
						automationMessageContent += `\nSummary: ${result.data.summary}`;
					}
					if (result.data.suggested_reply) {
						automationMessageContent += `\nSuggested Reply: ${result.data.suggested_reply}`;
					}
					if (Object.keys(result.data).length === 0 && result.message) {
						// If data is empty but there's a message, use that.
					} else if (Object.keys(result.data).length === 0 && !result.message) {
						automationMessageContent =
							"Automation completed, but no specific data returned.";
					}

					const automationSystemMessage = {
						id: `automation-${Date.now()}`,
						sender: "system", // Or 'assistant' if it's a direct reply
						text: automationMessageContent,
						isLoading: false,
					};

					setSessions((prevSessions) => {
						if (!prevSessions[activeSessionId]) return prevSessions;
						const currentHistory = prevSessions[activeSessionId].history || [];
						// Append the automation result as a new message
						const newHistory = [...currentHistory, automationSystemMessage];
						return {
							...prevSessions,
							[activeSessionId]: {
								...prevSessions[activeSessionId],
								history: newHistory,
							},
						};
					});
				} else if (result.status !== "success") {
					throw new Error(
						result.message ||
							result.error_details ||
							"Automation processing failed on backend.",
					);
				}
			} catch (error) {
				console.error(`API Hook: [${activeSessionId}] AUTOMATE error:`, error);
				const errorMessage = `Automation failed: ${error.message}`;
				setAutomationError(errorMessage);
				// Add error message to chat history
				setSessions((prevSessions) => {
					if (!prevSessions[activeSessionId]) return prevSessions;
					const currentHistory = prevSessions[activeSessionId].history || [];
					const errorMsg = {
						id: `error-automation-${Date.now()}`,
						sender: "system",
						text: `⚠️ ${errorMessage}`,
						isLoading: false,
					};
					return {
						...prevSessions,
						[activeSessionId]: {
							...prevSessions[activeSessionId],
							history: [...currentHistory, errorMsg],
						},
					};
				});
			} finally {
				setIsSubmitting(false);
				console.log(`API Hook: [${activeSessionId}] AUTOMATE finished.`);
			}
		},
		[activeSessionId, isSubmitting, setSessions, automateApiUrl],
	);

	return {
		isSubmitting,
		automationError,
		handleChatSubmit,
		handleAutomateConversation,
		setAutomationError,
		stopGeneration,
	};
}
