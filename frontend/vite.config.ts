import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
    plugins: [react()],
    build: {
      outDir: 'dist',
      // ensure assets are built with relative paths
      assetsDir: 'assets',
    },
    server: {
      proxy: {
        '/api': {
          target: '/',
          changeOrigin: true,
        },
      },
    },
  })