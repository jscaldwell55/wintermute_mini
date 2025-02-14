// index.tsx
import React from 'react';
import { createRoot } from 'react-dom/client'; // Import createRoot
import App from './app';
import './index.css';

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element not found');

const root = createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);