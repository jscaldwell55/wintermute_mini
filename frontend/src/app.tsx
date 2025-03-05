// src/app.tsx
import React, { useState, useEffect } from 'react';
// Import using exact path and case
import WintermuteInterface from './components/wintermute_interface';
import LoadingScreen from './components/LoadingScreen';

// Version indicator to track which app version is running
console.log('App component loading - Version 2.0');

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);
  // Generate a unique key to force remount of child components
  const [mountKey] = useState(`wintermute-${Date.now()}`);

  useEffect(() => {
    console.log('App component mounted with key:', mountKey);
    
    const timer = setTimeout(() => {
      console.log('Loading complete, showing WintermuteInterface');
      setLoading(false);
    }, 2500); // Simulate loading for 2.5 seconds

    return () => {
      console.log('App component unmounting, clearing timeout');
      clearTimeout(timer);
    }; 
  }, [mountKey]);

  // Force browser to reload script if a problem is detected
  useEffect(() => {
    if (!document.getElementById('force-reload')) {
      const script = document.createElement('script');
      script.id = 'force-reload';
      script.textContent = `
        // Force a hard reload if the app doesn't show interactions after 20 seconds
        setTimeout(() => {
          const historyContainer = document.querySelector('.overflow-y-auto');
          if (historyContainer && !historyContainer.textContent.includes('Wintermute:')) {
            console.log('Interaction display issue detected, reloading...');
            window.location.reload(true);
          }
        }, 20000);
      `;
      document.head.appendChild(script);
    }
  }, []);

  return (
    <>
      {loading ? (
        <LoadingScreen />
      ) : (
        // Use key to force a fresh mount of the component
        <WintermuteInterface key={mountKey} />
      )}
    </>
  );
};

export default App;