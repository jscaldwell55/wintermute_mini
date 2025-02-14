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
        outDir: '../dist', // Correct: Outputs to project root's dist/
        emptyOutDir: true, // Good practice: Cleans the output directory
        sourcemap: false,
        rollupOptions: {
            input: { // ADD THIS SECTION
              main: path.resolve(__dirname, 'index.html'), // Point to your HTML file
            },
            output: {
                manualChunks: {
                    vendor: [
                        'react',
                        'react-dom',
                        'lucide-react',
                    ],
                },
            },
        },
        chunkSizeWarningLimit: 1000,
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                drop_debugger: true,
            },
        },
    },
    optimizeDeps: {
        include: ['react', 'react-dom', 'lucide-react'],
    },
    server: {
        port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
        host: '0.0.0.0',
        strictPort: true,
    },
     root: './', //  Correct: frontend is the root for vite
});