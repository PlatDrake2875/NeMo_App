// frontend/src/components/guardrails/ConfigEditor.jsx
import PropTypes from "prop-types";
import Editor from "@monaco-editor/react";
import { useTheme } from "../../hooks/useTheme";
import { Skeleton } from "../ui/skeleton";
import { cn } from "../../lib/utils";

/**
 * ConfigEditor - Monaco editor wrapper for YAML and Colang files
 */
export function ConfigEditor({
  value,
  onChange,
  language = "yaml",
  readOnly = false,
  height = "100%",
  className,
}) {
  const { isDarkMode } = useTheme();

  const handleEditorChange = (newValue) => {
    if (onChange) {
      onChange(newValue || "");
    }
  };

  // Configure editor options
  const editorOptions = {
    minimap: { enabled: false },
    lineNumbers: "on",
    scrollBeyondLastLine: false,
    wordWrap: "on",
    tabSize: 2,
    insertSpaces: true,
    automaticLayout: true,
    readOnly,
    fontSize: 13,
    fontFamily: "'Fira Code', 'JetBrains Mono', Menlo, Monaco, 'Courier New', monospace",
    renderWhitespace: "selection",
    bracketPairColorization: { enabled: true },
    guides: {
      indentation: true,
      bracketPairs: true,
    },
    folding: true,
    foldingStrategy: "indentation",
  };

  // Loading component
  const LoadingComponent = () => (
    <div className="w-full h-full flex items-center justify-center">
      <Skeleton className="w-full h-full" />
    </div>
  );

  return (
    <div className={cn("border rounded-lg overflow-hidden", className)} style={{ height }}>
      <Editor
        height="100%"
        language={language}
        value={value}
        onChange={handleEditorChange}
        theme={isDarkMode ? "vs-dark" : "light"}
        options={editorOptions}
        loading={<LoadingComponent />}
        beforeMount={(monaco) => {
          // Register Colang language if not yaml
          if (language === "colang") {
            monaco.languages.register({ id: "colang" });
            monaco.languages.setMonarchTokensProvider("colang", {
              tokenizer: {
                root: [
                  // Comments
                  [/#.*$/, "comment"],
                  // Flow definitions
                  [/^define\s+/, "keyword"],
                  [/^flow\s+/, "keyword"],
                  [/^user\s+/, "keyword"],
                  [/^bot\s+/, "keyword"],
                  // Subflow calls
                  [/\$[a-zA-Z_][a-zA-Z0-9_]*/, "variable"],
                  // Strings
                  [/"[^"]*"/, "string"],
                  [/'[^']*'/, "string"],
                  // Keywords
                  [/\b(if|else|when|or|and|not|await|return|stop|continue)\b/, "keyword"],
                  // Actions
                  [/\b(execute|activate|send|log)\b/, "function"],
                ],
              },
            });
          }
        }}
      />
    </div>
  );
}

ConfigEditor.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func,
  language: PropTypes.oneOf(["yaml", "colang", "python"]),
  readOnly: PropTypes.bool,
  height: PropTypes.string,
  className: PropTypes.string,
};
