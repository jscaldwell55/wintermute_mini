// src/app.tsx
import React, { useState, useEffect } from 'react';
// Import the new component
import WintermuteInterfaceV2 from './components/WintermuteInterfaceV2';
import LoadingScreen from './components/LoadingScreen';

console.log('App loading with NEW V2 component - timestamp:', Date.now());

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      {loading ? (
        <LoadingScreen />
      ) : (
        <WintermuteInterfaceV2 />
      )}
    </>
  );
};

export default App;