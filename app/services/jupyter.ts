import {
    ServerConnection,
    KernelManager,
    KernelSpecManager,
    SessionManager,
    ContentsManager
} from '@jupyterlab/services';

/**
 * Configuration for Jupyter server connection.
 */
export interface JupyterConfig {
    baseUrl: string;
    token?: string;
    wsUrl?: string; // WebSocket URL, default is based on baseUrl
}

/**
 * Result from code execution.
 */
export interface CodeExecutionResult {
    success: boolean;
    outputs: any[];
    textOutput: string;
}

/**
 * Status of a Jupyter server connection.
 */
export interface JupyterStatus {
    connected: boolean;
    kernels?: any[];
    serverInfo?: any;
    error?: string;
}

/**
 * Service for connecting to and interacting with Jupyter servers.
 */
export class JupyterService {
    private _config: JupyterConfig | null = null;
    private _serverSettings: ServerConnection.ISettings | null = null;
    private _kernelManager: KernelManager | null = null;
    private _kernelSpecManager: KernelSpecManager | null = null;
    private _sessionManager: SessionManager | null = null;
    private _contentsManager: ContentsManager | null = null;
    private _defaultKernelPromise: Promise<string | undefined> | null = null;
    
    /**
     * Connect to a Jupyter server.
     */
    async connect(config: JupyterConfig): Promise<JupyterStatus> {
        try {
            // Clean up the base URL
            let baseUrl = config.baseUrl;
            
            // Remove trailing slash if present
            if (baseUrl.endsWith('/')) {
                baseUrl = baseUrl.slice(0, -1);
            }
            
            // If the URL contains "tree", make sure it's not part of the API endpoint
            if (baseUrl.includes('/tree')) {
                // Remove everything from "/tree" onwards
                baseUrl = baseUrl.split('/tree')[0];
            }
            
            // Ensure base URL ends with trailing slash for JupyterLab services
            if (!baseUrl.endsWith('/')) {
                baseUrl += '/';
            }
            
            this._config = {
                ...config,
                baseUrl
            };
            
            console.log('Connecting to Jupyter server at:', baseUrl);
            
            // Set up server connection settings with CORS support
            this._serverSettings = ServerConnection.makeSettings({
                baseUrl,
                wsUrl: config.wsUrl || this._getWebSocketUrl(baseUrl),
                token: config.token || '',
                appendToken: !!config.token,
                init: {
                    // Include credentials (cookies, authorization headers)
                    credentials: 'include',
                    // Set mode to cors explicitly
                    mode: 'cors',
                    cache: 'no-store',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            });
            
            // Create managers
            this._kernelManager = new KernelManager({ serverSettings: this._serverSettings });
            this._kernelSpecManager = new KernelSpecManager({ serverSettings: this._serverSettings });
            this._sessionManager = new SessionManager({ kernelManager: this._kernelManager, serverSettings: this._serverSettings });
            this._contentsManager = new ContentsManager({ serverSettings: this._serverSettings });
            
            // Set up default kernel promise
            this._defaultKernelPromise = this._findPythonKernel();
            
            // Try to connect to server and verify connection
            try {
                // Check connection by listing kernelspecs
                await this._kernelSpecManager.refreshSpecs();
                const specs = await this._kernelSpecManager.specs;
                
                if (!specs) {
                    throw new Error('No kernel specs available');
                }
                
                console.log('Successfully connected to Jupyter server');
                console.log('Available kernels:', Object.keys(specs.kernelspecs));
                
                return {
                    connected: true,
                    kernels: Object.keys(specs.kernelspecs),
                    serverInfo: specs
                };
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                console.error('Failed to retrieve kernel specs:', message);
                
                if (message.includes('CORS') || message.includes('NetworkError')) {
                    return {
                        connected: false,
                        error: `CORS error: Make sure the Jupyter server is running with proper CORS settings:
jupyter notebook --ServerApp.allow_origin="*" --ServerApp.allow_credentials=True --ServerApp.disable_check_xsrf=True`
                    };
                }
                
                return { 
                    connected: false,
                    error: `Connection error: ${message}`
                };
            }
        } catch (error) {
            console.error('Failed to connect to Jupyter server:', error);
            return { 
                connected: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
    
    /**
     * Execute code on a Jupyter kernel.
     */
    async executeCode(code: string, kernelName?: string): Promise<CodeExecutionResult> {
        if (!this._kernelManager || !this._serverSettings) {
            throw new Error('Not connected to a Jupyter server');
        }
        
        let kernel;
        const defaultKernel = await this._defaultKernelPromise;
        
        try {
            // Get kernel to use
            kernel = await this._getKernelByName(kernelName || defaultKernel || 'python');
            
            // Create future for executing the code
            const future = kernel.requestExecute({ code });
            
            // Collect outputs
            const outputs: any[] = [];
            let textOutput = '';
            
            future.onIOPub = (msg: any) => {
                const msgType = msg.header.msg_type;
                
                if (msgType === 'execute_result' || msgType === 'display_data') {
                    outputs.push(msg.content);
                    
                    if (msg.content.data && msg.content.data['text/plain']) {
                        textOutput += msg.content.data['text/plain'] + '\n';
                    }
                } else if (msgType === 'stream') {
                    outputs.push(msg.content);
                    textOutput += msg.content.text;
                } else if (msgType === 'error') {
                    outputs.push(msg.content);
                    textOutput += `ERROR: ${msg.content.ename}: ${msg.content.evalue}\n`;
                    textOutput += msg.content.traceback.join('\n');
                }
            };
            
            // Wait for execution to complete
            await future.done;
            
            return {
                success: !outputs.some(o => o.ename),
                outputs,
                textOutput
            };
        } catch (error) {
            console.error('Error executing code:', error);
            return {
                success: false,
                outputs: [],
                textOutput: `Execution error: ${error instanceof Error ? error.message : String(error)}`
            };
        }
    }
    
    /**
     * Disconnect from the Jupyter server.
     */
    async disconnect(): Promise<void> {
        if (this._sessionManager) {
            await this._sessionManager.shutdownAll();
        }
        
        if (this._kernelManager) {
            await this._kernelManager.shutdownAll();
        }
        
        this._config = null;
        this._serverSettings = null;
        this._kernelManager = null;
        this._kernelSpecManager = null;
        this._sessionManager = null;
        this._contentsManager = null;
        this._defaultKernelPromise = null;
    }
    
    /**
     * Get the WebSocket URL based on the base URL.
     */
    private _getWebSocketUrl(baseUrl: string): string {
        const url = new URL(baseUrl);
        url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
        return url.toString();
    }
    
    /**
     * Find a Python kernel to use as default.
     */
    private async _findPythonKernel(): Promise<string | undefined> {
        if (!this._kernelSpecManager) {
            return undefined;
        }
        
        try {
            const specs = await this._kernelSpecManager.specs;
            
            if (!specs) {
                return undefined;
            }
            
            // Try to find a Python kernel
            for (const name of Object.keys(specs.kernelspecs)) {
                if (name.toLowerCase().includes('python')) {
                    return name;
                }
            }
            
            // Fall back to default kernel
            return specs.default;
        } catch (error) {
            console.error('Error finding Python kernel:', error);
            return undefined;
        }
    }
    
    /**
     * Get a kernel by name, or create a new one if it doesn't exist.
     */
    private async _getKernelByName(name: string) {
        if (!this._kernelManager) {
            throw new Error('Not connected to a Jupyter server');
        }
        
        // Try to find an existing kernel first
        const running = await this._kernelManager.running();
        
        for (const model of running) {
            try {
                const kernel = await this._kernelManager.connectTo({ model });
                const info = await kernel.info;
                
                // Use an existing kernel if it matches our requirements
                if (model.name === name || 
                    (info.language_info && 
                     info.language_info.name && 
                     info.language_info.name.toLowerCase().includes('python'))) {
                    return kernel;
                }
            } catch (error) {
                // Skip this kernel if we can't connect to it
                console.warn('Failed to connect to kernel:', error);
                continue;
            }
        }
        
        // Start a new kernel if we couldn't find a suitable one
        return this._kernelManager.startNew({ name });
    }
}
