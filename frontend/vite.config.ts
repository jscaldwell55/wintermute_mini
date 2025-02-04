import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
    plugins: [
        react(),
        nodePolyfills({
        }),
    ],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            // crypto: 'crypto-browserify', // Alias crypto to crypto-browserify
        },
    },
    build: {
        outDir: 'dist',
        sourcemap: false,
        rollupOptions: {
            input: {
                main: path.resolve(__dirname, 'index.html'),
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