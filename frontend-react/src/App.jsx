import { useEffect, useState } from 'react';
import PortalPage from './components/PortalPage.jsx';
import ChatPage from './components/ChatPage.jsx';

const getRoute = () => {
  const hash = window.location.hash.replace('#', '');
  if (hash.startsWith('/chat')) {
    return 'chat';
  }
  return 'portal';
};

const App = () => {
  const [route, setRoute] = useState(getRoute());

  useEffect(() => {
    const handleHashChange = () => {
      setRoute(getRoute());
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return route === 'chat' ? <ChatPage /> : <PortalPage />;
};

export default App;
