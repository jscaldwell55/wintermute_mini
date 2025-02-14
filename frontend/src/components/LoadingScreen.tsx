// src/components/LoadingScreen.tsx
import React from 'react';

const LoadingScreen: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center w-full h-screen">
      <h1 className="text-4xl font-bold mb-8">wintermute ai system</h1>
      <div className="w-64 h-4 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-500 ease-in-out"
          style={{ width: '100%' }} // Animate to 100% width
        />
      </div>
      {/* REMOVE THIS LINE */}
    </div>
  );
};

export default LoadingScreen;