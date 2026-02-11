import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, MessageSquare, Paperclip, Send, X } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

const normalizeBackendMessage = (msg) => ({
  id: Date.now() + Math.random(),
  role: msg?.role ?? 'assistant',
  content: msg?.content ?? '',
});


const ChatPage = ({ embedded = false, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
  const createSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/chat/session`, { method: 'POST' });
      if (!res.ok) throw new Error(`Create session failed (${res.status})`);
      const data = await res.json();
      setSessionId(data.session_id);

      // Optional: show a greeting from backend once session exists
      setMessages([
        {
          id: Date.now(),
          role: 'assistant',
          content: 'Hi! Send a message to start, or upload a .h5ad file then ask me to annotate it.',
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages([
        {
          id: Date.now(),
          role: 'assistant',
          content:
            'I could not connect to the backend server. Make sure the backend is running on http://127.0.0.1:8000.',
        },
      ]);
    }
  };

  createSession();
}, []);


  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

 const API_BASE = "http://127.0.0.1:8000"; // hardcode for now to avoid env/proxy issues

const ensureSession = async () => {
  if (sessionId) return sessionId;

  const res = await fetch(`${API_BASE}/api/chat/session`, { method: "POST" });
  if (!res.ok) throw new Error(`Create session failed (${res.status})`);
  const data = await res.json();
  setSessionId(data.session_id);
  return data.session_id;
};

const handleSend = async () => {
  const text = inputValue.trim();
  if (!text) return;

  setMessages((prev) => [...prev, { id: Date.now(), role: "user", content: text }]);
  setInputValue("");
  setIsTyping(true);

  try {
    const sid = await ensureSession();

    // ✅ This is the ONLY message request. It ALWAYS has JSON body.
    const payload = { message: { role: "user", content: text } };
    console.log("SENDING PAYLOAD:", payload);

    const formData = new FormData();
    formData.append("message", text); // must match docs: message (string)

    const res = await fetch(`${API_BASE}/api/chat/${sid}/message`, {
      method: "POST",
      body: formData, // ✅ no JSON.stringify
  // ✅ IMPORTANT: do NOT set Content-Type manually for FormData
    });


    const raw = await res.text();
    console.log("RAW RESPONSE:", res.status, raw);

    if (!res.ok) throw new Error(raw);

    const data = JSON.parse(raw);
    setMessages((prev) => [
      ...prev,
      { id: Date.now() + 1, role: data.message.role, content: data.message.content },
    ]);
  } catch (err) {
    console.error(err);
    setMessages((prev) => [
      ...prev,
      { id: Date.now() + 1, role: "assistant", content: "Backend error. Check console + backend logs." },
    ]);
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
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`chat-message max-w-[80%] rounded-2xl p-4 ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg'
                    : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
                }`}
              >
                <p className="leading-relaxed">{msg.content}</p>
              </div>
            </div>
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

