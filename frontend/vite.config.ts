// frontend/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
    plugins: [
        react(),
        nodePolyfills({
            include: ['crypto'],
            globals: {
                Buffer: true,
                global: true,
                process: true,
            },
        }),
    ],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    build: {
        outDir: 'dist', // CORRECTED: Output to frontend/dist
        emptyOutDir: true,
        sourcemap: false, // Consider enabling sourcemaps for debugging, even in production
        rollupOptions: {
            input: {
              main: path.resolve(__dirname, 'index.html'), // Point to your HTML file
            },
            output: {
                manualChunks: {
                    vendor: [
                        'react',
                        'react-dom',
                        'lucide-react',
                        '@vapi-ai/web' // ADD VAPI HERE
                    ],
                },
            },
        },
        chunkSizeWarningLimit: 1000, // Consider removing this, or increasing it significantly.
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                drop_debugger: true,
            },
        },
    },
    optimizeDeps: {
        include: ['react', 'react-dom', 'lucide-react', '@vapi-ai/web'], // ADD VAPI HERE
    },
    server: {
        port:  3000,  // No need for process.env.PORT here, Vite's dev server doesn't use it.
        host: '0.0.0.0',
        strictPort: true,
        proxy: {  // Your proxy configuration is correct
          '/api': {
            target: 'http://localhost:8000',
            changeOrigin: true,
          },
        },
    },
    // No need for root: './' It defaults to the directory containing vite.config.ts
});