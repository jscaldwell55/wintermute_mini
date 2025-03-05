// src/app.tsx
import React, { useState, useEffect } from 'react';
import WintermuteInterface from './components/wintermute_interface';
import LoadingScreen from './components/LoadingScreen';

console.log('App component loading with integrated voice - Version 3.0');

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [mountKey] = useState(`wintermute-${Date.now()}`);

  useEffect(() => {
    console.log('App component mounted with key:', mountKey);
    
    const timer = setTimeout(() => {
      console.log('Loading complete, showing Wintermute interface');
      setLoading(false);
    }, 2000); // Reduced loading time for better UX

    return () => {
      console.log('App component unmounting, clearing timeout');
      clearTimeout(timer);
    }; 
  }, [mountKey]);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {loading ? (
        <LoadingScreen />
      ) : (
        <>
          {/* Header - removed the text */}
          <header className="bg-gray-800 p-4 shadow-md">
            <div className="container mx-auto">
              {/* Title text removed */}
            </div>
          </header>
          
          {/* Main content */}
          <main className="container mx-auto py-6 flex-1 flex flex-col">
            <WintermuteInterface key={`integrated-${mountKey}`} />
          </main>
          
          {/* Footer - removed the text */}
          <footer className="bg-gray-800 p-4">
            <div className="container mx-auto text-center text-gray-400 text-sm">
              {/* Footer text removed */}
            </div>
          </footer>
        </>
      )}
    </div>
  );
};

export default App;