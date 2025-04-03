import { app, BrowserWindow } from 'electron';
import path from 'path';

// Handle creating/removing shortcuts on Windows when installing/uninstalling
if (require('electron-squirrel-startup')) {
    app.quit();
}

// Debug output to see if dev server URL is being passed
console.log('VITE_DEV_SERVER_URL:', process.env.VITE_DEV_SERVER_URL);

// For development - hardcoded dev server URL
const DEV_SERVER_URL = 'http://localhost:5173';
const isDevelopment = process.env.NODE_ENV !== 'production';

const createWindow = () => {
    // Create the browser window
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, '../preload/preload.js'),
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    // Load the index.html of the app
    if (isDevelopment) {
        console.log('Loading from dev server:', DEV_SERVER_URL);
        mainWindow.loadURL(DEV_SERVER_URL);
        // Open DevTools in development mode
        mainWindow.webContents.openDevTools();
    } else {
        console.log('Loading from dist directory');
        // In production, load the built HTML file
        mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    }
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
