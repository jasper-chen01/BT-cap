import React, { useEffect, useState } from 'react';
import AuthModal from './components/AuthModal';
import ChatPage from './components/ChatPage';
import Header from './components/Header';
import LandingPage from './components/LandingPage';
import PortalPage from './components/PortalPage';
import ProfilePage from './components/ProfilePage';

const App = () => {
  const [user, setUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [route, setRoute] = useState('landing');

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');

      if (!user && (hash.startsWith('/portal') || hash.startsWith('/chat') || hash.startsWith('/profile'))) {
        window.location.hash = '';
        setRoute('landing');
        return;
      }

      if (hash.startsWith('/chat')) {
        setRoute('chat');
      } else if (hash.startsWith('/profile')) {
        setRoute('profile');
      } else if (hash.startsWith('/portal') || (user && hash === '')) {
        setRoute('portal');
      } else {
        setRoute('landing');
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();

    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [user]);

  const handleLogin = (userData) => {
    setUser(userData);
    setShowAuthModal(false);
    window.location.hash = '/portal';
  };

  const handleLogout = () => {
    setUser(null);
    window.location.hash = '';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30 flex flex-col">
      <Header
        user={user}
        onLoginClick={() => setShowAuthModal(true)}
        onLogout={handleLogout}
      />

      <main className="flex-1 flex flex-col relative z-10">
        {route === 'landing' && <LandingPage onGetStarted={() => setShowAuthModal(true)} />}
        {route === 'portal' && <PortalPage />}
        {route === 'chat' && <ChatPage />}
        {route === 'profile' && <ProfilePage user={user} />}
      </main>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onLogin={handleLogin}
      />
    </div>
  );
};

export default App;

