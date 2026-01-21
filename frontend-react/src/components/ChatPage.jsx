import { useEffect, useMemo, useRef, useState } from 'react';

const API_URL = 'http://localhost:8000/api';

const formatAssistantContent = (content) =>
  content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
    .replace(/`(.*?)`/g, '<code>$1</code>');

const ChatPage = () => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  const messagesContainerRef = useRef(null);
  const messageInputRef = useRef(null);

  const addMessage = (role, content, suggestions = null, annotationResults = null) => {
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role,
        content,
        suggestions,
        annotationResults,
      },
    ]);
  };

  const addWelcomeMessage = () => {
    const content = `👋 Hello! I'm your Brain Tumor Annotation Assistant.

I can help you:
- 📁 Upload and annotate single-cell glioma data
- 🔍 Analyze your data using our reference embeddings
- 📊 Provide detailed annotation results

You can upload a file by dragging it into the chat or clicking the upload button. How can I help you today?`;

    addMessage('assistant', content, [
      'Upload a file',
      'How does this work?',
      'Check system status',
    ]);
  };

  const createSession = async () => {
    try {
      const response = await fetch(`${API_URL}/chat/session`, {
        method: 'POST',
      });
      const session = await response.json();
      setSessionId(session.session_id);
      return session.session_id;
    } catch (error) {
      console.error('Error creating session:', error);
      return null;
    }
  };

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    return createSession();
  };

  useEffect(() => {
    createSession().then(() => addWelcomeMessage());
  }, []);

  useEffect(() => {
    if (messageInputRef.current) {
      messageInputRef.current.style.height = 'auto';
      messageInputRef.current.style.height = `${Math.min(
        messageInputRef.current.scrollHeight,
        150
      )}px`;
    }
  }, [inputValue]);

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith('.h5ad')) {
      alert('Please upload a .h5ad file');
      return;
    }

    addMessage('user', `📎 ${file.name}`);
    setIsTyping(true);

    try {
      const currentSession = await ensureSession();
      const formData = new FormData();
      formData.append('file', file);
      formData.append('message', `Upload file: ${file.name}`);

      const response = await fetch(`${API_URL}/chat/${currentSession}/message`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      addMessage(
        'assistant',
        result.message.content,
        result.suggestions,
        result.message.annotation_results
      );

      if (result.message.file_uploaded) {
        setUploadedFiles((prev) => [...prev, result.message.file_uploaded]);
      }
    } catch (error) {
      addMessage('assistant', `❌ Error uploading file: ${error.message}`);
    } finally {
      setIsTyping(false);
      if (event.target) event.target.value = '';
    }
  };

  const sendMessage = async () => {
    const message = inputValue.trim();
    if (!message && uploadedFiles.length === 0) return;

    if (message) {
      addMessage('user', message);
      setInputValue('');
    }

    setIsTyping(true);

    try {
      const currentSession = await ensureSession();
      const formData = new FormData();
      if (message) {
        formData.append('message', message);
      }

      const response = await fetch(`${API_URL}/chat/${currentSession}/message`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      addMessage(
        'assistant',
        result.message.content,
        result.suggestions,
        result.message.annotation_results
      );

      if (result.requires_action === 'annotate' && uploadedFiles.length > 0) {
        setTimeout(() => {
          annotateFile(uploadedFiles.length - 1);
        }, 1000);
      }
    } catch (error) {
      addMessage('assistant', `❌ Error: ${error.message}`);
    } finally {
      setIsTyping(false);
    }
  };

  const annotateFile = async (fileIndex = 0) => {
    addMessage('user', 'Annotate this file');
    setIsTyping(true);

    try {
      const currentSession = await ensureSession();
      const formData = new FormData();
      formData.append('file_index', fileIndex);
      formData.append('top_k', '10');
      formData.append('similarity_threshold', '0.7');

      const response = await fetch(`${API_URL}/chat/${currentSession}/annotate`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      addMessage(
        'assistant',
        result.message.content,
        result.suggestions,
        result.message.annotation_results
      );
    } catch (error) {
      addMessage('assistant', `❌ Error annotating: ${error.message}`);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSuggestion = (suggestion) => {
    if (suggestion.toLowerCase().includes('annotate')) {
      annotateFile();
      return;
    }

    setInputValue(suggestion);
    setTimeout(() => sendMessage(), 0);
  };

  const renderedMessages = useMemo(
    () =>
      messages.map((message) => (
        <div key={message.id} className={`message ${message.role}`}>
          <div className="message-content">
            {message.role === 'assistant' ? (
              <div
                dangerouslySetInnerHTML={{
                  __html: formatAssistantContent(message.content),
                }}
              />
            ) : (
              message.content
            )}

            {message.annotationResults && (
              <div className="annotation-results">
                <div className="annotation-summary">
                  <div className="annotation-stat">
                    <h4>{message.annotationResults.total_cells}</h4>
                    <p>Total Cells</p>
                  </div>
                  <div className="annotation-stat">
                    <h4>
                      {Object.keys(
                        message.annotationResults.annotation_counts || {}
                      ).length}
                    </h4>
                    <p>Unique Types</p>
                  </div>
                  <div className="annotation-stat">
                    <h4>
                      {(message.annotationResults.average_confidence * 100).toFixed(
                        1
                      )}
                      %
                    </h4>
                    <p>Avg Confidence</p>
                  </div>
                </div>
              </div>
            )}

            {message.suggestions?.length > 0 && (
              <div className="suggestions">
                {message.suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => handleSuggestion(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )),
    [messages]
  );

  return (
    <div className="chat-page">
      <div className="header">
        <h1>🧠 Brain Tumor Annotation Portal - AI Agent</h1>
        <div className="header-actions">
          <div className="status-indicator">
            <span className="status-dot" />
            <span>Online</span>
          </div>
          <a href="#/" className="link-button">
            Web Portal
          </a>
        </div>
      </div>

      <div className="chat-container">
        <div className="messages-container" ref={messagesContainerRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <h3>👋 Welcome!</h3>
              <p>Start a conversation to annotate your glioma single-cell data</p>
            </div>
          ) : (
            renderedMessages
          )}
          {isTyping && (
            <div className="message assistant" id="typingIndicator">
              <div className="message-content">
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="input-container">
          <div className="file-upload-area">
            <label className="file-upload-button">
              📎 Upload
              <input type="file" accept=".h5ad" onChange={handleFileUpload} />
            </label>
          </div>
          <div className="input-wrapper">
            <textarea
              ref={messageInputRef}
              className="message-input"
              placeholder="Type your message..."
              rows={1}
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage();
                }
              }}
            />
          </div>
          <button className="send-button" onClick={sendMessage}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;

