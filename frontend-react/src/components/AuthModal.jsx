import React, { useState } from 'react';
import { Activity, Brain } from 'lucide-react';
import Button from './ui/Button';
import Input from './ui/Input';

const API_URL = 'http://localhost:8000/api';

const AuthModal = ({ isOpen, onClose, onLogin }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isSignUp ? 'auth/signup' : 'auth/signin';
      const payload = isSignUp ? { name, email, password } : { email, password };
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || 'Authentication failed');
      }

      const data = await response.json();
      onLogin({
        userId: data.user_id,
        name: data.name,
        email: data.email,
        avatar: null,
      });
      onClose();
    } catch (err) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-300">
        <div className="p-8">
          <div className="text-center mb-8">
            <div className="mx-auto w-12 h-12 bg-indigo-500/10 rounded-xl flex items-center justify-center mb-4">
              <Brain className="text-indigo-400" size={24} />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">
              {isSignUp ? 'Create Account' : 'Welcome Back'}
            </h2>
            <p className="text-slate-400 text-sm">
              {isSignUp
                ? 'Join the research network'
                : 'Enter your credentials to access the portal'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignUp && (
              <Input
                label="Full Name"
                placeholder="Dr. Jane Doe"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            )}
            <Input
              label="Email Address"
              type="email"
              placeholder="name@institution.edu"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
            <Input
              label="Password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            {error && <p className="text-sm text-red-400">{error}</p>}

            <Button
              type="submit"
              className="w-full mt-6"
              disabled={loading || !email || !password || (isSignUp && !name)}
            >
              {loading ? (
                <Activity className="animate-spin" />
              ) : isSignUp ? (
                'Sign Up'
              ) : (
                'Sign In'
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-sm text-slate-400 hover:text-indigo-400 transition-colors"
            >
              {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;

