import React, { useState } from 'react';
import { Download, AlertCircle, CheckCircle2, Loader2, Lock } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { API_BASE_URL } from '../lib/api-config';

/**
 * Dialog component for downloading models from HuggingFace
 * Features: validation, gated model support, progress tracking
 */
export function ModelDownloadDialog({ open, onOpenChange, onDownloadComplete }) {
  const [modelId, setModelId] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [modelMetadata, setModelMetadata] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadStage, setDownloadStage] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const resetState = () => {
    setModelId('');
    setHfToken('');
    setModelMetadata(null);
    setIsValidating(false);
    setIsDownloading(false);
    setDownloadProgress(0);
    setDownloadStage('');
    setError(null);
    setSuccess(false);
  };

  const handleClose = () => {
    if (!isDownloading) {
      resetState();
      onOpenChange(false);
    }
  };

  const handleValidate = async () => {
    if (!modelId.trim()) {
      setError('Please enter a model ID');
      return;
    }

    setError(null);
    setIsValidating(true);
    setModelMetadata(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/models/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId.trim(),
          token: hfToken || null,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Validation failed');
      }

      const metadata = await response.json();
      setModelMetadata(metadata);
    } catch (err) {
      setError(err.message);
      setModelMetadata(null);
    } finally {
      setIsValidating(false);
    }
  };

  const handleDownload = async () => {
    if (!modelMetadata) {
      setError('Please validate the model first');
      return;
    }

    if (modelMetadata.is_gated && !hfToken) {
      setError('This model is gated and requires a HuggingFace token');
      return;
    }

    setError(null);
    setSuccess(false);
    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadStage('Initializing...');

    try {
      const response = await fetch(`${API_BASE_URL}/api/models/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId.trim(),
          token: hfToken || null,
        }),
      });

      if (!response.ok) {
        throw new Error('Download request failed');
      }

      // Read SSE stream for progress updates
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));

              if (data.stage === 'error') {
                setError(data.error || 'Download failed');
                setIsDownloading(false);
                return;
              }

              setDownloadProgress(data.progress || 0);
              setDownloadStage(data.message || '');

              if (data.stage === 'complete') {
                setSuccess(true);
                setIsDownloading(false);
                // Notify parent to refresh model list
                if (onDownloadComplete) {
                  onDownloadComplete(modelId.trim());
                }
                // Auto-close after 2 seconds
                setTimeout(() => {
                  handleClose();
                }, 2000);
                return;
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data:', parseError);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Download failed');
      setIsDownloading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Download Model from HuggingFace
          </DialogTitle>
          <DialogDescription>
            Enter a HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct) to download
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Model ID Input */}
          <div className="space-y-2">
            <Label htmlFor="model-id">Model ID</Label>
            <div className="flex gap-2">
              <Input
                id="model-id"
                placeholder="meta-llama/Llama-3.1-8B-Instruct"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                disabled={isDownloading || success}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isValidating) {
                    handleValidate();
                  }
                }}
              />
              <Button
                onClick={handleValidate}
                disabled={isValidating || isDownloading || success}
                variant="outline"
              >
                {isValidating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Validate'
                )}
              </Button>
            </div>
          </div>

          {/* Model Metadata Display */}
          {modelMetadata && (
            <div className="space-y-3 rounded-lg border p-3 bg-muted/50">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Model Information</span>
                {modelMetadata.is_gated && (
                  <Badge variant="warning" className="gap-1">
                    <Lock className="h-3 w-3" />
                    Gated
                  </Badge>
                )}
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Size:</span>
                  <span className="ml-2 font-medium">{modelMetadata.size_gb} GB</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Downloads:</span>
                  <span className="ml-2 font-medium">
                    {modelMetadata.downloads.toLocaleString()}
                  </span>
                </div>
              </div>

              {modelMetadata.pipeline_tag && (
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="secondary">{modelMetadata.pipeline_tag}</Badge>
                  {modelMetadata.tags?.slice(0, 3).map((tag) => (
                    <Badge key={tag} variant="outline">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* HF Token Input (shown if model is gated) */}
          {modelMetadata?.is_gated && (
            <div className="space-y-2">
              <Label htmlFor="hf-token">
                HuggingFace Token
                <span className="text-destructive ml-1">*</span>
              </Label>
              <Input
                id="hf-token"
                type="password"
                placeholder="hf_..."
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                disabled={isDownloading || success}
              />
              <p className="text-xs text-muted-foreground">
                This model requires authentication. Get your token from{' '}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  HuggingFace settings
                </a>
              </p>
            </div>
          )}

          {/* Progress Bar */}
          {isDownloading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{downloadStage}</span>
                <span className="font-medium">{downloadProgress}%</span>
              </div>
              <Progress value={downloadProgress} className="h-2" />
            </div>
          )}

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Success Alert */}
          {success && (
            <Alert>
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription>
                Model downloaded successfully! It will appear in the model list.
              </AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isDownloading}>
            {success ? 'Close' : 'Cancel'}
          </Button>
          <Button
            onClick={handleDownload}
            disabled={!modelMetadata || isDownloading || success}
          >
            {isDownloading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Downloading...
              </>
            ) : success ? (
              <>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Downloaded
              </>
            ) : (
              <>
                <Download className="mr-2 h-4 w-4" />
                Download Model
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
