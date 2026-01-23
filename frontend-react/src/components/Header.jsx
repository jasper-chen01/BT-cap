import React, { useState } from 'react';
import { Brain, LogOut, Settings, User } from 'lucide-react';
import Button from './ui/Button';

const Header = ({ user, onLoginClick, onLogout }) => {
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur supports-[backdrop-filter]:bg-slate-950/60">
      <div className="container flex h-16 items-center justify-between px-4 md:px-8 mx-auto">
        <div className="flex items-center gap-2">
          <div className="bg-gradient-to-tr from-indigo-500 to-cyan-500 p-2 rounded-lg">
            <Brain className="text-white" size={20} />
          </div>
          <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400 hidden sm:inline-block">
            NeuroAnnotate
          </span>
        </div>

        <div className="flex items-center gap-4">
          {user ? (
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowProfileMenu(!showProfileMenu)}
                className="flex items-center gap-3 p-1.5 pr-4 rounded-full bg-slate-800/50 hover:bg-slate-800 border border-slate-700 transition-all"
              >
                <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-sm font-bold text-white">
                  {user.name.charAt(0)}
                </div>
                <span className="text-sm font-medium text-slate-200 hidden md:block">
                  {user.name}
                </span>
              </button>

              {showProfileMenu && (
                <div className="absolute right-0 mt-2 w-56 bg-slate-900 border border-slate-700 rounded-xl shadow-xl py-1 animate-in fade-in slide-in-from-top-2">
                  <div className="px-4 py-3 border-b border-slate-800">
                    <p className="text-sm font-medium text-white">{user.name}</p>
                    <p className="text-xs text-slate-500 truncate">{user.email}</p>
                  </div>
                  <a
                    href="#/profile"
                    className="w-full text-left px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white flex items-center gap-2"
                    onClick={() => setShowProfileMenu(false)}
                  >
                    <User size={14} /> Profile
                  </a>
                  <button
                    type="button"
                    className="w-full text-left px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white flex items-center gap-2"
                  >
                    <Settings size={14} /> Settings
                  </button>
                  <div className="border-t border-slate-800 my-1"></div>
                  <button
                    type="button"
                    onClick={onLogout}
                    className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                  >
                    <LogOut size={14} /> Sign Out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-4">
              <button
                type="button"
                className="text-sm font-medium text-slate-300 hover:text-white transition-colors hidden md:block"
              >
                Documentation
              </button>
              <Button onClick={onLoginClick} variant="primary" className="px-4 py-2 text-sm">
                Sign In
              </Button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;

