import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: '',
    root: 'app',
    build: {
        outDir: '../dist/renderer',
    },
    resolve: {
        alias: {
            '@': resolve(__dirname, './app'),
        },
    },
});
