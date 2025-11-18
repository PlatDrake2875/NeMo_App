# Frontend Development Guide

Complete guide to the React frontend architecture and implementation.

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Component Architecture](#component-architecture)
5. [State Management](#state-management)
6. [Custom Hooks](#custom-hooks)
7. [API Integration](#api-integration)
8. [Styling](#styling)
9. [Component Reference](#component-reference)
10. [Best Practices](#best-practices)

## Overview

The frontend is a React 19 single-page application built with Vite for fast development and optimized production builds.

### Key Features

- **Streaming Chat**: Real-time SSE response streaming
- **Session Management**: Local storage-based persistence
- **Dark/Light Theme**: User preference with localStorage
- **Markdown Rendering**: GitHub Flavored Markdown support
- **Agent Selection**: Modal-based agent switcher
- **Document Management**: Upload and view PDFs
- **Responsive Design**: Works on desktop and mobile

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.0 | UI framework |
| Vite | 6.2 | Build tool, dev server |
| react-markdown | Latest | Markdown rendering |
| remark-gfm | Latest | GitHub Flavored Markdown |
| CSS Modules | - | Component-scoped styling |

**Why These Choices?**
- **React 19**: Latest features, concurrent rendering
- **Vite**: Extremely fast HMR, optimized builds
- **No Redux**: Simple state management with hooks
- **CSS Modules**: Scoped styles, no naming conflicts

## Project Structure

```
frontend/
├── src/
│   ├── main.jsx                  # Entry point
│   ├── App.jsx                   # Root component
│   │
│   ├── components/               # React components
│   │   ├── Header.jsx           # Top navigation
│   │   ├── Sidebar.jsx          # Session list, controls
│   │   ├── ChatInterface.jsx    # Main chat area
│   │   ├── ChatHistory.jsx      # Message display
│   │   ├── ChatForm.jsx         # Input form
│   │   ├── MarkdownMessage.jsx  # Message renderer
│   │   ├── AgentSelector.jsx    # Agent selection modal
│   │   └── DocumentViewer.jsx   # Document manager
│   │
│   ├── hooks/                   # Custom hooks
│   │   ├── useChatSessions.js  # Session management
│   │   └── useTheme.js         # Theme toggle
│   │
│   ├── styles/                  # CSS files
│   │   ├── App.css
│   │   ├── Header.css
│   │   └── ...
│   │
│   └── assets/                  # Static assets
│
├── public/                      # Public files
│   └── favicon.ico
│
├── index.html                   # HTML template
├── vite.config.js              # Vite configuration
├── package.json                # Dependencies
├── .eslintrc.cjs              # ESLint config
└── Dockerfile                  # Container image
```

## Component Architecture

### Component Hierarchy

```
App (Root State Container)
├── Header
│   ├── Logo
│   └── Theme Toggle
│
├── Sidebar
│   ├── New Session Button
│   ├── Session List
│   │   └── Session Item (multiple)
│   ├── Model Selector
│   ├── Agent Selector Button
│   └── Document Viewer Button
│
└── ChatInterface
    ├── ChatHistory
    │   └── MarkdownMessage (multiple)
    └── ChatForm
        ├── Textarea Input
        ├── RAG Toggle
        └── Submit Button

Modals (Rendered at root level)
├── AgentSelector
│   └── Agent Cards
└── DocumentViewer
    ├── Upload Form
    └── Document List
```

### Data Flow

```
User Action (e.g., Send Message)
       ↓
ChatForm Component
       ↓
handleSendMessage() in App.jsx
       ↓
API Call (fetch/SSE)
       ↓
State Update (setMessages)
       ↓
ChatHistory Re-render
       ↓
MarkdownMessage Renders New Message
```

## State Management

### No Global State Library

Instead of Redux/Context, we use:
1. **useState** for component state
2. **Custom hooks** for shared logic
3. **Props** for parent-child communication
4. **localStorage** for persistence

### App-Level State

**File**: `frontend/src/App.jsx:15-45`

```jsx
export default function App() {
  // Session management
  const {
    sessions,
    currentSessionId,
    createSession,
    deleteSession,
    updateSession,
    setCurrentSessionId
  } = useChatSessions();

  // Current session state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Settings state
  const [selectedModel, setSelectedModel] = useState('gemma3:4b-it-q4_K_M');
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [useRag, setUseRag] = useState(false);

  // UI state
  const [showAgentSelector, setShowAgentSelector] = useState(false);
  const [showDocumentViewer, setShowDocumentViewer] = useState(false);

  // Theme
  const { theme, toggleTheme } = useTheme();

  // ... component logic
}
```

### Component-Level State

**Example**: ChatForm component

```jsx
export default function ChatForm({ onSubmit, isLoading }) {
  const [inputValue, setInputValue] = useState('');
  const [useRag, setUseRag] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    onSubmit({
      query: inputValue,
      useRag: useRag
    });

    setInputValue('');  // Clear after submit
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* ... */}
    </form>
  );
}
```

### State Persistence

**Session Data** (useChatSessions hook):

```jsx
// Save to localStorage on every change
useEffect(() => {
  localStorage.setItem('chatSessions', JSON.stringify(sessions));
}, [sessions]);

// Load from localStorage on mount
const [sessions, setSessions] = useState(() => {
  const stored = localStorage.getItem('chatSessions');
  return stored ? JSON.parse(stored) : [];
});
```

**Theme Preference** (useTheme hook):

```jsx
const [theme, setTheme] = useState(() => {
  return localStorage.getItem('theme') || 'light';
});

useEffect(() => {
  localStorage.setItem('theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
}, [theme]);
```

## Custom Hooks

### useChatSessions Hook

**File**: `frontend/src/hooks/useChatSessions.js:10-180`

**Purpose**: Manage chat sessions with localStorage persistence

```jsx
export function useChatSessions() {
  // State
  const [sessions, setSessions] = useState(() => {
    const stored = localStorage.getItem('chatSessions');
    return stored ? JSON.parse(stored) : [];
  });

  const [currentSessionId, setCurrentSessionId] = useState(null);

  // Persist to localStorage
  useEffect(() => {
    localStorage.setItem('chatSessions', JSON.stringify(sessions));
  }, [sessions]);

  // Create new session
  const createSession = () => {
    const newSession = {
      id: Date.now().toString(),
      name: `Chat ${sessions.length + 1}`,
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    setSessions([newSession, ...sessions]);
    setCurrentSessionId(newSession.id);

    return newSession;
  };

  // Delete session
  const deleteSession = (id) => {
    setSessions(sessions.filter(s => s.id !== id));

    if (currentSessionId === id) {
      setCurrentSessionId(null);
    }
  };

  // Update session
  const updateSession = (id, updates) => {
    setSessions(sessions.map(s =>
      s.id === id
        ? { ...s, ...updates, updatedAt: new Date().toISOString() }
        : s
    ));
  };

  // Rename session
  const renameSession = (id, newName) => {
    updateSession(id, { name: newName });
  };

  // Get current session
  const currentSession = sessions.find(s => s.id === currentSessionId);

  return {
    sessions,
    currentSessionId,
    currentSession,
    createSession,
    deleteSession,
    updateSession,
    renameSession,
    setCurrentSessionId
  };
}
```

**Usage**:
```jsx
function App() {
  const { sessions, createSession, deleteSession } = useChatSessions();

  return (
    <Sidebar
      sessions={sessions}
      onNewSession={createSession}
      onDeleteSession={deleteSession}
    />
  );
}
```

### useTheme Hook

**File**: `frontend/src/hooks/useTheme.js:8-45`

**Purpose**: Theme toggle with persistence

```jsx
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  useEffect(() => {
    localStorage.setItem('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(current => current === 'light' ? 'dark' : 'light');
  };

  return { theme, toggleTheme };
}
```

**CSS Variables**:
```css
:root[data-theme="light"] {
  --bg-primary: #ffffff;
  --text-primary: #000000;
  --border-color: #e0e0e0;
}

:root[data-theme="dark"] {
  --bg-primary: #1a1a1a;
  --text-primary: #ffffff;
  --border-color: #333333;
}
```

## API Integration

### REST API Calls

**Example: Fetch Models**

```jsx
const [models, setModels] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  async function fetchModels() {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/models');

      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }

      const data = await response.json();
      setModels(data.models);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  fetchModels();
}, []);
```

**Example: Upload File**

```jsx
const handleUpload = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8000/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const result = await response.json();
    console.log(`Uploaded: ${result.filename}, ${result.chunks_added} chunks`);
  } catch (error) {
    console.error('Upload error:', error);
  }
};
```

### Server-Sent Events (SSE)

**File**: `frontend/src/App.jsx:85-145`

**Chat Streaming Implementation**:

```jsx
const handleSendMessage = async (query) => {
  // Add user message immediately
  const userMessage = { sender: 'user', text: query };
  const newMessages = [...messages, userMessage];
  setMessages(newMessages);

  // Create placeholder for bot response
  const botMessage = { sender: 'bot', text: '', streaming: true };
  setMessages([...newMessages, botMessage]);

  // Build API URL
  const apiUrl = new URL('http://localhost:8000/api/chat');

  // Make POST request to initiate SSE stream
  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        model: selectedModel,
        agent_name: selectedAgent,
        use_rag: useRag,
        history: messages.map(m => ({ sender: m.sender, text: m.text }))
      })
    });

    // Read SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      // Decode chunk
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          if (data.token) {
            // Append token to bot message
            setMessages(current => {
              const updated = [...current];
              const lastMsg = updated[updated.length - 1];
              lastMsg.text += data.token;
              return updated;
            });
          } else if (data.status === 'done') {
            // Mark streaming complete
            setMessages(current => {
              const updated = [...current];
              const lastMsg = updated[updated.length - 1];
              lastMsg.streaming = false;
              return updated;
            });
          } else if (data.error) {
            throw new Error(data.error);
          }
        }
      }
    }
  } catch (error) {
    console.error('Chat error:', error);
    // Show error message
    setMessages(current => [
      ...current,
      { sender: 'bot', text: `Error: ${error.message}`, error: true }
    ]);
  }
};
```

**Key Points**:
1. **Immediate UI Update**: User message shows instantly
2. **Streaming Placeholder**: Bot message created before streaming starts
3. **Token Accumulation**: Each SSE token appends to message
4. **Error Handling**: Errors shown as bot messages
5. **State Updates**: React state updates trigger re-renders

## Styling

### CSS Modules Approach

**Component CSS** (`ChatForm.css`):
```css
.chat-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1rem;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-color);
}

.textarea {
  width: 100%;
  min-height: 60px;
  padding: 0.75rem;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  font-family: inherit;
  resize: vertical;
}

.submit-button {
  align-self: flex-end;
  padding: 0.75rem 1.5rem;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.submit-button:hover {
  background: var(--primary-color-dark);
}

.submit-button:disabled {
  background: var(--disabled-color);
  cursor: not-allowed;
}
```

**Import in Component**:
```jsx
import './ChatForm.css';

export default function ChatForm() {
  return (
    <form className="chat-form">
      <textarea className="textarea" />
      <button className="submit-button">Send</button>
    </form>
  );
}
```

### Theme Variables

**File**: `frontend/src/styles/App.css`

```css
:root {
  /* Light theme */
  --bg-primary: #ffffff;
  --bg-secondary: #f5f5f5;
  --text-primary: #000000;
  --text-secondary: #666666;
  --border-color: #e0e0e0;
  --primary-color: #007bff;
  --primary-color-dark: #0056b3;
  --disabled-color: #cccccc;
  --error-color: #dc3545;
  --success-color: #28a745;
}

:root[data-theme="dark"] {
  /* Dark theme */
  --bg-primary: #1a1a1a;
  --bg-secondary: #2a2a2a;
  --text-primary: #ffffff;
  --text-secondary: #aaaaaa;
  --border-color: #333333;
  --primary-color: #0d6efd;
  --primary-color-dark: #0a58ca;
  --disabled-color: #555555;
  --error-color: #dc3545;
  --success-color: #28a745;
}
```

### Responsive Design

```css
/* Mobile first */
.sidebar {
  width: 100%;
  height: auto;
}

/* Tablet and up */
@media (min-width: 768px) {
  .sidebar {
    width: 250px;
    height: 100vh;
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .sidebar {
    width: 300px;
  }
}
```

## Component Reference

### App Component

**File**: `frontend/src/App.jsx`

**Responsibilities**:
- Root state management
- API integration
- Component coordination
- Session management

**Key Methods**:
- `handleSendMessage()` - Send chat message
- `handleNewSession()` - Create new session
- `handleDeleteSession()` - Delete session
- `handleSelectAgent()` - Choose agent

### ChatInterface Component

**File**: `frontend/src/components/ChatInterface.jsx`

**Props**:
```jsx
interface ChatInterfaceProps {
  messages: Array<{ sender: string, text: string }>;
  onSendMessage: (query: string) => void;
  isLoading: boolean;
  selectedModel: string;
  useRag: boolean;
  onToggleRag: (value: boolean) => void;
}
```

**Features**:
- Message display
- Input form
- Auto-scroll
- Loading states

### ChatHistory Component

**File**: `frontend/src/components/ChatHistory.jsx`

**Props**:
```jsx
interface ChatHistoryProps {
  messages: Array<{ sender: string, text: string }>;
  isLoading: boolean;
}
```

**Features**:
- Scroll container
- Message list rendering
- Auto-scroll to bottom
- Loading indicator

**Auto-scroll Implementation**:
```jsx
const messagesEndRef = useRef(null);
const [autoScroll, setAutoScroll] = useState(true);

useEffect(() => {
  if (autoScroll && messagesEndRef.current) {
    messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }
}, [messages, autoScroll]);

const handleScroll = (e) => {
  const { scrollTop, scrollHeight, clientHeight } = e.target;
  const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
  setAutoScroll(isAtBottom);
};

return (
  <div className="chat-history" onScroll={handleScroll}>
    {messages.map((msg, i) => (
      <MarkdownMessage key={i} message={msg} />
    ))}
    <div ref={messagesEndRef} />
  </div>
);
```

### MarkdownMessage Component

**File**: `frontend/src/components/MarkdownMessage.jsx`

**Props**:
```jsx
interface MarkdownMessageProps {
  message: {
    sender: 'user' | 'bot';
    text: string;
    streaming?: boolean;
    error?: boolean;
  };
}
```

**Implementation**:
```jsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function MarkdownMessage({ message }) {
  return (
    <div className={`message message-${message.sender}`}>
      <div className="message-header">
        <span className="sender">{message.sender === 'user' ? 'You' : 'Assistant'}</span>
      </div>
      <div className="message-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.text}
        </ReactMarkdown>
        {message.streaming && <span className="cursor">▋</span>}
      </div>
    </div>
  );
}
```

**Supported Markdown**:
- Headings (# ## ###)
- Lists (- * 1.)
- Code blocks (\`\`\`)
- Inline code (\`)
- Links
- Bold/Italic
- Tables (GFM)
- Strikethrough (GFM)

### AgentSelector Component

**File**: `frontend/src/components/AgentSelector.jsx`

**Props**:
```jsx
interface AgentSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectAgent: (agentId: string) => void;
  selectedAgent: string | null;
}
```

**Features**:
- Modal overlay
- Agent grid display
- Agent metadata (icon, name, description)
- Selection state

**Implementation Pattern**:
```jsx
export default function AgentSelector({ isOpen, onClose, onSelectAgent }) {
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    if (isOpen) {
      fetch('http://localhost:8000/api/agents/metadata')
        .then(res => res.json())
        .then(data => setAgents(data.agents));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h2>Select Agent</h2>
        <div className="agent-grid">
          {agents.map(agent => (
            <div
              key={agent.id}
              className="agent-card"
              onClick={() => {
                onSelectAgent(agent.id);
                onClose();
              }}
            >
              <div className="agent-icon">{agent.icon}</div>
              <h3>{agent.name}</h3>
              <p>{agent.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

## Best Practices

### Component Design

✅ **Do**: Small, focused components
```jsx
// Good: Single responsibility
function ChatMessage({ message }) {
  return <div className="message">{message.text}</div>;
}
```

❌ **Don't**: Monolithic components
```jsx
// Bad: Too many responsibilities
function ChatApp() {
  // Hundreds of lines handling everything
}
```

### State Management

✅ **Do**: Lift state to common ancestor
```jsx
function App() {
  const [messages, setMessages] = useState([]);

  return (
    <>
      <ChatHistory messages={messages} />
      <ChatForm onSend={(msg) => setMessages([...messages, msg])} />
    </>
  );
}
```

❌ **Don't**: Prop drilling through many levels
```jsx
// Bad: Passing through unnecessary intermediaries
<A prop={x}>
  <B prop={x}>
    <C prop={x}>
      <D prop={x} /> {/* Finally uses it */}
    </C>
  </B>
</A>
```

### Performance

✅ **Do**: Memoize expensive computations
```jsx
const sortedMessages = useMemo(() => {
  return messages.sort((a, b) => a.timestamp - b.timestamp);
}, [messages]);
```

✅ **Do**: Use keys for lists
```jsx
{messages.map(msg => (
  <Message key={msg.id} message={msg} />
))}
```

❌ **Don't**: Use index as key for dynamic lists
```jsx
{messages.map((msg, i) => (
  <Message key={i} message={msg} /> {/* Bad */}
))}
```

### Error Handling

✅ **Do**: Handle errors gracefully
```jsx
try {
  const response = await fetch('/api/chat');
  if (!response.ok) throw new Error('Request failed');
  // ...
} catch (error) {
  setError(error.message);
  console.error('Chat error:', error);
}
```

### Accessibility

✅ **Do**: Semantic HTML and ARIA labels
```jsx
<button
  aria-label="Send message"
  disabled={isLoading}
>
  {isLoading ? 'Sending...' : 'Send'}
</button>
```

## Related Documentation

- [Architecture Overview](./ARCHITECTURE.md) - Frontend architecture
- [Development Guide](./DEVELOPMENT.md) - Local frontend development
- [API Reference](./API-REFERENCE.md) - Backend API integration
- [Troubleshooting](./TROUBLESHOOTING.md) - Frontend issues
