// Preload script to expose specific Node.js APIs to the renderer process
import { contextBridge, ipcRenderer } from 'electron';

// Expose a limited API to the renderer process
contextBridge.exposeInMainWorld('electron', {
    // Expose methods for IPC communication
    send: (channel: string, data: any) => {
    // Whitelist channels to ensure security
        const validChannels = ['graph-operation', 'save-graph', 'load-graph'];
        if (validChannels.includes(channel)) {
            ipcRenderer.send(channel, data);
        }
    },
    receive: (channel: string, callback: Function) => {
    // Whitelist channels to ensure security
        const validChannels = ['graph-result', 'save-result', 'load-result'];
        if (validChannels.includes(channel)) {
            // Remove the event listener if it exists to avoid duplicates
            ipcRenderer.removeAllListeners(channel);
            // Add a new listener
            ipcRenderer.on(channel, (_, ...args) => callback(...args));
        }
    }
});
