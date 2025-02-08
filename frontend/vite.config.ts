// vite.config.ts (FINAL CORRECT VERSION)
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
            '@': path.resolve(__dirname, './src'), // Correct alias for src
        },
    },
    build: {
        outDir: '../dist', // Output to the project root's dist/
        sourcemap: false,
        rollupOptions: {
            // NO input option here. Let Vite handle index.html
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
     root: '../', // Set the root to the project root.  CRUCIAL.
});