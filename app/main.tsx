import React from 'react';
import ReactDOM from 'react-dom/client';
import { MantineProvider } from '@mantine/core';
import '@mantine/core/styles.css';
import 'reactflow/dist/base.css';
import App from './App';
import './style.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <MantineProvider defaultColorScheme="dark">
            <App />
        </MantineProvider>
    </React.StrictMode>
);
