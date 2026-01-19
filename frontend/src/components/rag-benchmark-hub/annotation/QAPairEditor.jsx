import { useState } from "react";
import PropTypes from "prop-types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../ui/card";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import { Textarea } from "../../ui/textarea";
import { Label } from "../../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import { Edit2, Save, X } from "lucide-react";

/**
 * Editor component for a single Q&A pair
 */
export function QAPairEditor({ pair, annotation, onEdit, onSetDifficulty }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedPair, setEditedPair] = useState({ ...pair });

  const handleSave = () => {
    onEdit(editedPair);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedPair({ ...pair });
    setIsEditing(false);
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "approved":
        return <Badge className="bg-green-500">Approved</Badge>;
      case "rejected":
        return <Badge variant="destructive">Rejected</Badge>;
      default:
        return <Badge variant="outline">Pending</Badge>;
    }
  };

  const getDifficultyBadge = (difficulty) => {
    switch (difficulty) {
      case "easy":
        return <Badge className="bg-blue-500">Easy</Badge>;
      case "hard":
        return <Badge className="bg-orange-500">Hard</Badge>;
      default:
        return <Badge variant="secondary">Medium</Badge>;
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Q&A Pair</CardTitle>
          <div className="flex items-center gap-2">
            {getStatusBadge(annotation.status)}
            {getDifficultyBadge(annotation.difficulty)}
            {!isEditing && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsEditing(true)}
              >
                <Edit2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Question */}
        <div className="space-y-2">
          <Label>Question</Label>
          {isEditing ? (
            <Textarea
              value={editedPair.query || editedPair.question || ""}
              onChange={(e) =>
                setEditedPair((prev) => ({ ...prev, query: e.target.value }))
              }
              rows={2}
            />
          ) : (
            <div className="p-3 bg-muted rounded-md text-sm">
              {pair.query || pair.question}
            </div>
          )}
        </div>

        {/* Answer */}
        <div className="space-y-2">
          <Label>Expected Answer</Label>
          {isEditing ? (
            <Textarea
              value={editedPair.ground_truth || editedPair.expected_answer || ""}
              onChange={(e) =>
                setEditedPair((prev) => ({ ...prev, ground_truth: e.target.value }))
              }
              rows={4}
            />
          ) : (
            <div className="p-3 bg-muted rounded-md text-sm whitespace-pre-wrap">
              {pair.ground_truth || pair.expected_answer}
            </div>
          )}
        </div>

        {/* Difficulty selector */}
        <div className="space-y-2">
          <Label>Difficulty</Label>
          <Select
            value={annotation.difficulty}
            onValueChange={onSetDifficulty}
          >
            <SelectTrigger className="w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="easy">Easy</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="hard">Hard</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Metadata */}
        {pair.metadata && Object.keys(pair.metadata).length > 0 && (
          <div className="space-y-2">
            <Label className="text-muted-foreground">Metadata</Label>
            <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
              {Object.entries(pair.metadata).map(([key, value]) => (
                <div key={key}>
                  <span className="font-medium">{key}:</span> {String(value)}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Edit actions */}
        {isEditing && (
          <div className="flex items-center gap-2 justify-end">
            <Button variant="outline" size="sm" onClick={handleCancel}>
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button size="sm" onClick={handleSave}>
              <Save className="h-4 w-4 mr-1" />
              Save Changes
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

QAPairEditor.propTypes = {
  pair: PropTypes.object.isRequired,
  annotation: PropTypes.object.isRequired,
  onEdit: PropTypes.func.isRequired,
  onSetDifficulty: PropTypes.func.isRequired,
};

export default QAPairEditor;
