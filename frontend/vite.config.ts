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
        outDir: '../dist', // Output to project root's dist/
        sourcemap: false,
        rollupOptions: {
            // No need for manual input if index.html is in the root.
            // Vite will find it automatically
            output: {
                manualChunks: {
                    vendor: [
                        'react',
                        'react-dom',
                        'recharts',
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
        include: ['react', 'react-dom', 'recharts', 'lucide-react'],
    },
    server: {
        port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
        host: '0.0.0.0',
        strictPort: true,
    },
    root: './', // Add this to point to project root
});