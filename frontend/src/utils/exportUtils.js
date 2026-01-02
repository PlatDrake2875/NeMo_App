import { jsPDF } from "jspdf";

/**
 * Export conversation as enhanced JSON with metadata
 * @param {Object} session - Session object with name and history
 * @param {string} sessionId - Session ID
 * @param {string} model - Model used in the session
 */
export function exportAsJSON(session, sessionId, model = null) {
  const exportData = {
    exportVersion: "1.0",
    exportDate: new Date().toISOString(),
    sessionInfo: {
      id: sessionId,
      name: session.name || sessionId,
      createdAt: session.history[0]?.timestamp || new Date().toISOString(),
      messageCount: session.history.length,
      model: model
    },
    messages: session.history.map((msg) => ({
      id: msg.id,
      sender: msg.sender,
      text: msg.text,
      timestamp: msg.timestamp
    }))
  };

  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  const downloadName = session.name
    ? session.name.replace(/[^a-z0-9]/gi, "_").toLowerCase()
    : sessionId;
  link.download = `${downloadName}_conversation.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export conversation as formatted PDF
 * @param {Object} session - Session object with name and history
 * @param {string} sessionId - Session ID
 * @param {string} model - Model used in the session
 */
export function exportAsPDF(session, sessionId, model = null) {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  const maxWidth = pageWidth - 2 * margin;
  let yPosition = margin;

  // Helper to add page if needed
  const checkPageBreak = (height) => {
    if (yPosition + height > pageHeight - margin) {
      doc.addPage();
      yPosition = margin;
      return true;
    }
    return false;
  };

  // Title
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  const title = session.name || sessionId;
  doc.text(title, margin, yPosition);
  yPosition += 10;

  // Metadata
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100, 100, 100);
  doc.text(`Exported: ${new Date().toLocaleString()}`, margin, yPosition);
  yPosition += 5;
  doc.text(`Messages: ${session.history.length}`, margin, yPosition);
  yPosition += 5;
  if (model) {
    doc.text(`Model: ${model}`, margin, yPosition);
    yPosition += 5;
  }
  yPosition += 10;

  // Divider
  doc.setDrawColor(200, 200, 200);
  doc.line(margin, yPosition, pageWidth - margin, yPosition);
  yPosition += 10;

  // Messages
  session.history.forEach((msg) => {
    const isUser = msg.sender === "user";
    const senderLabel = isUser ? "You" : "Assistant";
    const timestamp = msg.timestamp
      ? new Date(msg.timestamp).toLocaleTimeString()
      : "";

    // Sender header
    checkPageBreak(15);
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(isUser ? 59 : 100, isUser ? 130 : 100, isUser ? 246 : 100);
    doc.text(`${senderLabel}`, margin, yPosition);
    if (timestamp) {
      doc.setFontSize(9);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(150, 150, 150);
      doc.text(timestamp, pageWidth - margin - 30, yPosition);
    }
    yPosition += 6;

    // Message content
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(0, 0, 0);

    // Split text into lines
    const lines = doc.splitTextToSize(msg.text, maxWidth);
    lines.forEach((line) => {
      checkPageBreak(5);
      doc.text(line, margin, yPosition);
      yPosition += 5;
    });

    yPosition += 8;
  });

  // Save
  const downloadName = session.name
    ? session.name.replace(/[^a-z0-9]/gi, "_").toLowerCase()
    : sessionId;
  doc.save(`${downloadName}_conversation.pdf`);
}

/**
 * Parse and validate imported JSON conversation
 * @param {string} jsonString - JSON string to parse
 * @returns {Object} Parsed and validated conversation data
 */
export function parseImportedConversation(jsonString) {
  const data = JSON.parse(jsonString);

  // Handle both old format (direct array) and new format (with metadata)
  if (Array.isArray(data)) {
    // Old format: direct array of messages
    return {
      sessionInfo: {
        name: null,
        importedAt: new Date().toISOString()
      },
      messages: data.map((msg, index) => ({
        id: msg.id || `imported-${index}`,
        sender: msg.sender || "user",
        text: msg.text || "",
        timestamp: msg.timestamp
      }))
    };
  }

  // New format with metadata
  if (data.messages && Array.isArray(data.messages)) {
    return {
      sessionInfo: {
        name: data.sessionInfo?.name || null,
        importedAt: new Date().toISOString(),
        originalExportDate: data.exportDate
      },
      messages: data.messages.map((msg, index) => ({
        id: `imported-${Date.now()}-${index}`,
        sender: msg.sender || "user",
        text: msg.text || "",
        timestamp: msg.timestamp
      }))
    };
  }

  throw new Error("Invalid conversation format");
}
