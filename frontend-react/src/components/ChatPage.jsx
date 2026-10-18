import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, MessageSquare, Paperclip, Send, X } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import TracePanel from './TracePanel';
import { API_BASE, apiRequest, normalizeChatMessage } from '../lib/reasoningApi';

export function ChatMessage({ message: msg }) {
  return (
    <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`chat-message max-w-[80%] rounded-2xl p-4 ${
        msg.role === 'user'
          ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg'
          : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
      }`}>
        <p className="leading-relaxed whitespace-pre-line">{msg.content}</p>
        {msg.role === 'assistant' && msg.traces?.length > 0 && <TracePanel traces={msg.traces} />}
      </div>
    </div>
  );
}

const ChatPage = ({ embedded = false, onClose, analysisSummaryPath = '' }) => {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const createSession = async () => {
      try {
        const data = await apiRequest('/chat/session', { method: 'POST' });
        setSessionId(data.session_id);
        setMessages([normalizeChatMessage({
          role: 'assistant',
          content: 'Hi! Send a message to start, or upload a .h5ad file then ask me to annotate it.',
        })]);
      } catch (err) {
        console.error(err);
        setMessages([normalizeChatMessage({
          role: 'assistant',
          content: `I could not connect to the backend server. Make sure the backend is running on ${API_BASE}.`,
        })]);
      }
    };
    createSession();
  }, []);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    const data = await apiRequest('/chat/session', { method: 'POST' });
    setSessionId(data.session_id);
    return data.session_id;
  };

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text) return;

    setMessages((prev) => [...prev, normalizeChatMessage({ role: 'user', content: text })]);
    setInputValue('');
    setIsTyping(true);

    try {
      const sid = await ensureSession();
      const formData = new FormData();
      formData.append('message', text);
      formData.append('analysis_summary_path', analysisSummaryPath);
      const data = await apiRequest(`/chat/${sid}/message`, { method: 'POST', body: formData });
      setMessages((prev) => [...prev, normalizeChatMessage(data.message)]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, normalizeChatMessage({
        role: 'assistant', content: 'Backend error. Check console + backend logs.',
      })]);
    } finally {
      setIsTyping(false);
    }
  };
  const containerClassName = embedded
    ? 'h-full w-full flex flex-col p-4'
    : 'max-w-5xl mx-auto h-[calc(100vh-5rem)] flex flex-col p-4 md:p-8';

  return (
    <div className={containerClassName}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <MessageSquare className="text-cyan-400" />
          Research Assistant
        </h2>
        {embedded ? (
          <button
            type="button"
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white transition-colors"
            aria-label="Close chat"
          >
            <X size={20} />
          </button>
        ) : (
          <a href="#/">
            <Button variant="secondary" className="px-4 py-2 text-sm" icon={ArrowRight}>
              Back to Portal
            </Button>
          </a>
        )}
      </div>

      <Card className="flex-1 mb-4 flex flex-col overflow-hidden bg-slate-900/50 border-slate-700/50">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="chat-message bg-slate-800 rounded-2xl rounded-tl-none p-4 border border-slate-700 flex gap-2">
                <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-75" />
                <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce delay-150" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </Card>

      <div className="relative">
        <textarea
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              handleSend();
            }
          }}
          placeholder="Ask about your data..."
          className="w-full bg-slate-800/80 backdrop-blur text-white rounded-xl border border-slate-700 p-4 pr-32 focus:ring-2 focus:ring-indigo-500 resize-none shadow-xl"
          rows="3"
        />
        <div className="absolute bottom-4 right-4 flex gap-2">
          <button type="button" className="p-2 text-slate-400 hover:text-white transition-colors">
            <Paperclip size={20} />
          </button>
          <button
            type="button"
            onClick={handleSend}
            disabled={!inputValue.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded-lg transition-colors"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;

