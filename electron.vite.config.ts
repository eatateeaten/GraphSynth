import { defineConfig } from 'electron-vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
    // Configuration for the Electron main process
    main: {
        build: {
            outDir: 'dist/main', // Output directory for main process build
            rollupOptions: {
                input: {
                    main: resolve(__dirname, 'electron/main.ts')
                }
            }
        },
    },
    // Configuration for the preload script
    preload: {
        build: {
            outDir: 'dist/preload', // Output directory for preload build
            rollupOptions: {
                input: {
                    preload: resolve(__dirname, 'electron/preload.ts')
                }
            }
        },
    },
    // Configuration for the renderer process
    renderer: {
        root: 'app', // Root directory for the renderer (where app/index.html is located)
        build: {
            outDir: 'dist/renderer', // Output directory for renderer build
            rollupOptions: {
                input: {
                    index: resolve(__dirname, 'app/index.html')
                }
            }
        },
        plugins: [react()], // Enable React support for the renderer
    },
});
