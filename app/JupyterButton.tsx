import { useState } from 'react';
import { Button, TextInput, Group, Modal, Alert } from '@mantine/core';
import { useStore } from './store';
import type { JupyterConfig } from './services/jupyter';

export function JupyterButton() {
    const [serverUrl, setServerUrl] = useState<string>('http://localhost:8888/');
    const [modalOpen, setModalOpen] = useState<boolean>(false);
    const [connectionError, setConnectionError] = useState<string | undefined>(undefined);
    
    const {
        jupyter,
        connectToJupyter,
        disconnectFromJupyter
    } = useStore();
    
    const { isConnecting, status } = jupyter;
    
    const handleConnect = async () => {
        setConnectionError(undefined);
        
        try {
            // Parse the URL to extract the token if present
            const url = new URL(serverUrl);
            const token = url.searchParams.get('token') || undefined;
            
            // Remove token from URL before connecting
            if (token) {
                url.searchParams.delete('token');
            }
            
            const config: JupyterConfig = {
                baseUrl: url.toString(),
                token
            };
            
            const result = await connectToJupyter(config);
            
            if (result.connected) {
                setModalOpen(false);
            } else if (result.error) {
                setConnectionError(result.error);
            }
        } catch (error) {
            console.error('Invalid URL format:', error);
            setConnectionError(`Invalid URL format: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    
    const handleDisconnect = async () => {
        await disconnectFromJupyter();
    };
    
    const openModal = () => {
        setConnectionError(undefined);
        setModalOpen(true);
    };
    
    const closeModal = () => setModalOpen(false);
    
    return (
        <>
            {status?.connected ? (
                <Button 
                    onClick={handleDisconnect} 
                    variant="outline" 
                    color="teal"
                >
                    Connected
                </Button>
            ) : (
                <Button 
                    onClick={openModal} 
                    variant="outline"
                >
                    Connect...
                </Button>
            )}
            
            <Modal
                opened={modalOpen}
                onClose={closeModal}
                title="Connect to Jupyter Notebook"
                size="lg"
            >
                <TextInput
                    label="Server URL with token"
                    placeholder="http://localhost:8888/?token=abc123"
                    description="Paste the complete URL from your Jupyter notebook, including the token"
                    value={serverUrl}
                    onChange={(e) => setServerUrl(e.target.value)}
                    style={{ width: '100%' }}
                    mb="md"
                />
                
                {connectionError && (
                    <Alert 
                        color="red" 
                        title="Connection Error" 
                        mb="md"
                        styles={{
                            message: {
                                whiteSpace: 'pre-wrap'
                            }
                        }}
                    >
                        {connectionError}
                    </Alert>
                )}
                
                <Group justify="flex-end">
                    <Button variant="outline" onClick={closeModal}>Cancel</Button>
                    <Button onClick={handleConnect} loading={isConnecting}>Connect</Button>
                </Group>
            </Modal>
        </>
    );
} 