import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        sourcemap: false,
        minify: 'terser',
        target: 'es2015',
        base: './',
        rollupOptions: {
            output: {
                manualChunks: undefined,
            },
        },
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    server: {
        port: process.env.PORT ? parseInt(process.env.PORT) : 3000,
        host: true,
        proxy: {
            '/api': {
                target: '/',
                changeOrigin: true,
            },
        },
    }
})