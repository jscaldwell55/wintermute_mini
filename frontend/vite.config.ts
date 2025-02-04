import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'), // Now points to frontend/index.html
      },
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            if (id.includes('recharts')) {
              return 'recharts';
            }
            return 'vendor';
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