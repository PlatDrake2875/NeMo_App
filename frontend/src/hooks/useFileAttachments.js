// frontend/src/hooks/useFileAttachments.js
import { useState, useCallback } from "react";

// Allowed file types for attachments
const ALLOWED_FILE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "application/pdf",
  "text/plain",
  "text/markdown",
  "text/csv",
  "application/json",
];

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_FILES = 5;

/**
 * Hook for managing file attachments in chat
 * @returns {Object} Attachment state and handlers
 */
export function useFileAttachments() {
  const [attachments, setAttachments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  /**
   * Validate a file before adding
   * @param {File} file - The file to validate
   * @returns {string|null} Error message or null if valid
   */
  const validateFile = useCallback((file) => {
    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      return `File type not allowed: ${file.type}`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large: ${(file.size / 1024 / 1024).toFixed(2)}MB (max ${MAX_FILE_SIZE / 1024 / 1024}MB)`;
    }
    return null;
  }, []);

  /**
   * Add files to attachments
   * @param {FileList|File[]} files - Files to add
   */
  const addFiles = useCallback(
    (files) => {
      const fileArray = Array.from(files);
      const errors = [];
      const validFiles = [];

      // Check total file count
      if (attachments.length + fileArray.length > MAX_FILES) {
        setUploadError(`Maximum ${MAX_FILES} files allowed`);
        return;
      }

      fileArray.forEach((file) => {
        const error = validateFile(file);
        if (error) {
          errors.push(`${file.name}: ${error}`);
        } else {
          // Check for duplicates
          const isDuplicate = attachments.some(
            (att) => att.file.name === file.name && att.file.size === file.size
          );
          if (!isDuplicate) {
            validFiles.push({
              id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              file,
              preview: file.type.startsWith("image/")
                ? URL.createObjectURL(file)
                : null,
              status: "pending", // pending, uploading, uploaded, error
            });
          }
        }
      });

      if (errors.length > 0) {
        setUploadError(errors.join("; "));
      } else {
        setUploadError(null);
      }

      if (validFiles.length > 0) {
        setAttachments((prev) => [...prev, ...validFiles]);
      }
    },
    [attachments, validateFile]
  );

  /**
   * Remove an attachment by ID
   * @param {string} attachmentId - The ID of the attachment to remove
   */
  const removeAttachment = useCallback((attachmentId) => {
    setAttachments((prev) => {
      const attachment = prev.find((att) => att.id === attachmentId);
      // Revoke object URL to prevent memory leaks
      if (attachment?.preview) {
        URL.revokeObjectURL(attachment.preview);
      }
      return prev.filter((att) => att.id !== attachmentId);
    });
    setUploadError(null);
  }, []);

  /**
   * Clear all attachments
   */
  const clearAttachments = useCallback(() => {
    // Revoke all object URLs
    attachments.forEach((att) => {
      if (att.preview) {
        URL.revokeObjectURL(att.preview);
      }
    });
    setAttachments([]);
    setUploadError(null);
  }, [attachments]);

  /**
   * Upload attachments to the server
   * @param {string} apiBaseUrl - The base URL for the API
   * @returns {Promise<Array>} Array of uploaded file info
   */
  const uploadAttachments = useCallback(
    async (apiBaseUrl) => {
      if (attachments.length === 0) return [];

      setIsUploading(true);
      setUploadError(null);

      const uploadedFiles = [];

      try {
        for (const attachment of attachments) {
          if (attachment.status === "uploaded") {
            uploadedFiles.push(attachment);
            continue;
          }

          // Update status to uploading
          setAttachments((prev) =>
            prev.map((att) =>
              att.id === attachment.id ? { ...att, status: "uploading" } : att
            )
          );

          const formData = new FormData();
          formData.append("file", attachment.file);

          const response = await fetch(`${apiBaseUrl}/api/chat/attachments`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
              errorData.detail || `Failed to upload ${attachment.file.name}`
            );
          }

          const result = await response.json();

          // Update status to uploaded
          setAttachments((prev) =>
            prev.map((att) =>
              att.id === attachment.id
                ? { ...att, status: "uploaded", uploadedInfo: result }
                : att
            )
          );

          uploadedFiles.push({ ...attachment, uploadedInfo: result });
        }

        return uploadedFiles;
      } catch (error) {
        console.error("Error uploading attachments:", error);
        setUploadError(error.message);

        // Mark failed attachments
        setAttachments((prev) =>
          prev.map((att) =>
            att.status === "uploading" ? { ...att, status: "error" } : att
          )
        );

        throw error;
      } finally {
        setIsUploading(false);
      }
    },
    [attachments]
  );

  /**
   * Get attachments ready for sending with a message
   * @returns {Array} Array of attachment metadata for the message
   */
  const getAttachmentsForMessage = useCallback(() => {
    return attachments.map((att) => ({
      id: att.id,
      name: att.file.name,
      type: att.file.type,
      size: att.file.size,
      preview: att.preview,
      uploadedInfo: att.uploadedInfo,
    }));
  }, [attachments]);

  return {
    attachments,
    isUploading,
    uploadError,
    addFiles,
    removeAttachment,
    clearAttachments,
    uploadAttachments,
    getAttachmentsForMessage,
    hasAttachments: attachments.length > 0,
    maxFiles: MAX_FILES,
    allowedTypes: ALLOWED_FILE_TYPES,
  };
}
