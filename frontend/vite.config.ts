import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      // To exclude specific polyfills, add them to this list.
      exclude: [],
      // Whether to polyfill specific globals.
      globals: {
        Buffer: true, // can also be 'build', 'dev', or false
        global: true,
        process: true,
      },
      // Whether to polyfill `node:` protocol imports.
      protocolImports: true,
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist', // Output directory (will be frontend/dist)
    sourcemap: false, // Consider 'hidden' or false for production to reduce bundle size
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'), // Points to frontend/index.html
      },
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            // Example: Put all modules from the 'node_modules/recharts' directory into a 'recharts' chunk
            if (id.includes('recharts')) {
              return 'recharts';
            }
            return 'vendor'; // Default chunk for all other node_modules
          }
        },
      },
    },
  },
  server: {
    port: process.env.PORT ? parseInt(process.env.PORT) : 5173,
    host: true,
  },
});