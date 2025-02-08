// src/app.tsx
import React, { useState, useEffect } from 'react';
import WintermuteInterface from './components/wintermute_interface';
import LoadingScreen from './components/LoadingScreen'; // Import the LoadingScreen

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2500); // Simulate loading for 2.5 seconds

    return () => clearTimeout(timer); // Cleanup on unmount
  }, []);

  return (
    <>
      {loading ? <LoadingScreen /> : <WintermuteInterface />}
    </>
  );
};

export default App;