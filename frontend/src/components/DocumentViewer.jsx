// src/components/DocumentViewer.jsx
import { useEffect, useMemo, useState } from "react";
import styles from "./DocumentViewer.module.css";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const DOCUMENTS_API_URL = `${API_BASE_URL}/api/documents`;

export function DocumentViewer({ onBackToChat }) {
	const [allDocuments, setAllDocuments] = useState([]); // Store all fetched docs
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState(null);
	const [selectedSource, setSelectedSource] = useState(""); // State for selected source

	useEffect(() => {
		const fetchDocuments = async () => {
			setIsLoading(true);
			setError(null);
			try {
				const response = await fetch(DOCUMENTS_API_URL);
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
				const data = await response.json();
				console.log("Fetched data from /api/documents:", data);

				if (data && Array.isArray(data.documents)) {
					// Sort documents primarily by source, then by page number
					const sortedDocs = data.documents.sort((a, b) => {
						const sourceA =
							a.metadata?.original_filename || a.metadata?.source || "";
						const sourceB =
							b.metadata?.original_filename || b.metadata?.source || "";
						if (sourceA < sourceB) return -1;
						if (sourceA > sourceB) return 1;
						const pageA = a.metadata?.page ?? -1;
						const pageB = b.metadata?.page ?? -1;
						return pageA - pageB;
					});
					setAllDocuments(sortedDocs); // Store all sorted documents
				} else {
					console.error("Invalid data format received:", data);
					throw new Error("Invalid data format received from server.");
				}
			} catch (err) {
				console.error("Error fetching documents:", err);
				setError(err.message || "Failed to fetch documents.");
				setAllDocuments([]);
			} finally {
				setIsLoading(false);
			}
		};

		fetchDocuments();
	}, []);

	// --- Get unique source names for the dropdown ---
	const availableSources = useMemo(() => {
		if (!allDocuments || allDocuments.length === 0) {
			return [];
		}
		const sources = new Set();
		allDocuments.forEach((doc) => {
			sources.add(
				doc.metadata?.original_filename ||
					doc.metadata?.source ||
					"Unknown Source",
			);
		});
		return Array.from(sources); // Convert Set to Array
	}, [allDocuments]);

	// --- Filter documents based on selected source ---
	const filteredDocuments = useMemo(() => {
		if (!selectedSource) {
			return []; // Return empty array if no source is selected
		}
		return allDocuments.filter((doc) => {
			const sourceName =
				doc.metadata?.original_filename ||
				doc.metadata?.source ||
				"Unknown Source";
			return sourceName === selectedSource;
		});
	}, [allDocuments, selectedSource]);

	// --- Handle dropdown change ---
	const handleSourceChange = (event) => {
		setSelectedSource(event.target.value);
	};

	// --- Handle back to chat with defensive check ---
	const handleBackClick = () => {
		if (typeof onBackToChat === "function") {
			onBackToChat();
		} else {
			console.error("onBackToChat is not a function:", onBackToChat);
		}
	};

	return (
		<div className={styles.documentViewer}>
			<header className={styles.viewerHeader}>
				{/* Header Content: Title, Dropdown, Back Button */}
				<div className={styles.headerLeft}>
					<h1>Uploaded Document Chunks</h1>
				</div>
				<div className={styles.headerControls}>
					{/* Dropdown for selecting source */}
					{availableSources.length > 0 && !isLoading && (
						<div className={styles.sourceSelectorContainer}>
							<label
								htmlFor="source-select"
								className={styles.sourceSelectLabel}
							>
								Select Document:
							</label>
							<select
								id="source-select"
								value={selectedSource}
								onChange={handleSourceChange}
								className={styles.sourceSelect}
								aria-label="Select document source to view chunks"
							>
								<option value="">-- Select a Document --</option>
								{availableSources.map((source) => (
									<option key={source} value={source}>
										{/* Display only the filename part */}
										{source.includes("/")
											? source.substring(source.lastIndexOf("/") + 1)
											: source}
									</option>
								))}
							</select>
						</div>
					)}
					<button
						type="button"
						onClick={handleBackClick}
						className={styles.backButton}
					>
						&larr; Back to Chat
					</button>
				</div>
			</header>

			<div className={styles.viewerContent}>
				{isLoading && (
					<p className={styles.loadingMessage}>Loading documents...</p>
				)}
				{error && <p className={styles.errorMessage}>Error: {error}</p>}

				{/* Show message if loading is done but no source is selected */}
				{!isLoading &&
					!error &&
					!selectedSource &&
					availableSources.length > 0 && (
						<p className={styles.noDocumentsMessage}>
							Please select a document from the dropdown above.
						</p>
					)}

				{/* Show message if no documents exist at all */}
				{!isLoading && !error && availableSources.length === 0 && (
					<p className={styles.noDocumentsMessage}>
						No documents found in the vector store.
					</p>
				)}

				{/* Render the list only if a source is selected */}
				{!isLoading && !error && selectedSource && (
					<ul className={styles.documentList}>
						{filteredDocuments.map((doc, index) => {
							console.log(`Rendering document index ${index}:`, doc);

							let displayContent = "[Content not available]";
							if (typeof doc.content === "string") {
								displayContent = doc.content;
							} else if (typeof doc.page_content === "string") {
								displayContent = doc.page_content;
							}

							return (
								<li
									key={doc.id || `doc-${index}`}
									className={styles.documentItem}
								>
									<div className={styles.documentMetadata}>
										{doc.metadata?.page !== undefined && (
											<span>
												<strong>Page:</strong> {doc.metadata.page + 1}
											</span>
										)}
										<span>
											{" "}
											| <strong>Chunk ID:</strong> {doc.id}
										</span>
									</div>
									<pre className={styles.documentContent}>{displayContent}</pre>
								</li>
							);
						})}
					</ul>
				)}
			</div>
		</div>
	);
}
