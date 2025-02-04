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
        outDir: 'dist',
        sourcemap: false,
        rollupOptions: {
            input: {
                main: path.resolve(__dirname, 'index.html'),
            },
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
        // Add chunk size optimizations
        chunkSizeWarningLimit: 1000,
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                drop_debugger: true,
            },
        },
    },
    // Optimize memory usage
    optimizeDeps: {
        include: ['react', 'react-dom', 'recharts', 'lucide-react'],
    },
    server: {
        port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
        host: '0.0.0.0',
        strictPort: true,
    },
});