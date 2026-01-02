import PropTypes from "prop-types";
import { useState } from "react";
import { ChevronDown, FileText, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { ScrollArea } from "../ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from "../ui/accordion";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "../ui/dialog";

export function DocumentSourceList({
  documents,
  onDeleteDocument,
  isLoading
}) {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Group documents by source
  const groupedDocuments = documents.reduce((acc, doc) => {
    const source = doc.metadata?.original_filename || doc.metadata?.source || "Unknown";
    if (!acc[source]) {
      acc[source] = {
        filename: source,
        chunks: [],
        chunkingMethod: doc.metadata?.chunking_method || "unknown"
      };
    }
    acc[source].chunks.push(doc);
    return acc;
  }, {});

  const sources = Object.values(groupedDocuments);

  const handleDeleteClick = (source) => {
    setDocumentToDelete(source);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!documentToDelete) return;
    setIsDeleting(true);
    try {
      await onDeleteDocument(documentToDelete.filename);
      setDeleteDialogOpen(false);
      setDocumentToDelete(null);
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setIsDeleting(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Documents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (sources.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Documents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No documents uploaded yet</p>
            <p className="text-sm mt-1">Upload a document to get started</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Documents ({sources.length})</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[400px]">
            <Accordion type="multiple" className="w-full">
              {sources.map((source) => (
                <AccordionItem key={source.filename} value={source.filename}>
                  <div className="flex items-center px-4">
                    <AccordionTrigger className="flex-1 py-3">
                      <div className="flex items-center gap-3 text-left">
                        <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
                        <div className="min-w-0">
                          <p className="font-medium truncate text-sm">
                            {source.filename.includes("/")
                              ? source.filename.substring(source.filename.lastIndexOf("/") + 1)
                              : source.filename}
                          </p>
                          <div className="flex gap-2 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {source.chunks.length} chunks
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {source.chunkingMethod}
                            </Badge>
                          </div>
                        </div>
                      </div>
                    </AccordionTrigger>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteClick(source)}
                      className="h-8 w-8 text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  <AccordionContent className="px-4 pb-3">
                    <div className="space-y-2 ml-8">
                      {source.chunks.slice(0, 5).map((chunk, index) => (
                        <div
                          key={chunk.id || index}
                          className="p-2 bg-muted rounded text-xs"
                        >
                          <div className="flex justify-between text-muted-foreground mb-1">
                            {chunk.metadata?.page !== undefined && (
                              <span>Page {chunk.metadata.page + 1}</span>
                            )}
                            <span className="truncate ml-2">ID: {chunk.id}</span>
                          </div>
                          <p className="line-clamp-2">
                            {chunk.content || chunk.page_content || "[No content]"}
                          </p>
                        </div>
                      ))}
                      {source.chunks.length > 5 && (
                        <p className="text-xs text-muted-foreground text-center">
                          + {source.chunks.length - 5} more chunks
                        </p>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{documentToDelete?.filename}&quot;?
              This will remove all {documentToDelete?.chunks.length} chunks from the vector store.
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
              disabled={isDeleting}
            >
              {isDeleting ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

DocumentSourceList.propTypes = {
  documents: PropTypes.array.isRequired,
  onDeleteDocument: PropTypes.func.isRequired,
  isLoading: PropTypes.bool
};
