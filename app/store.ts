/**
 * This store implements a global state management pattern using Zustand.
 * Rather than prop-drilling or managing distributed state across components,
 * we centralize our application state here for predictable data flow and updates.
 * 
 * The store maintains two parallel graph representations:
 * 1. CheckerGraph: Domain logic layer
 *        - Validates tensor shapes and parameter constraints
 *        - Manages node connectivity and data flow
 *        - Source of truth for computational graph structure
 * 
 * 2. React Flow: Presentation layer
 *        - Handles node positioning and visual layout
 *        - Manages UI state (selection, dragging, etc)
 *        - Displays validation errors from CheckerGraph
 * 
 * Operations follow a consistent pattern:
 * 1. Validate operation against CheckerGraph (shape checking, valid connections)
 * 2. On success: update React Flow state
 * 3. On failure: propagate error to UI layer
 */

import { create } from 'zustand';
import type { FlowNode, FlowEdge } from './types';
import { NodeType } from '../OpCompiler/types';
import { GRID_SIZE } from './config';

import { Graph as CompilerGraph } from '../OpCompiler/graph';
import { JupyterService, type JupyterConfig, type JupyterStatus, type CodeExecutionResult } from './services/jupyter';

interface NodeConfig {
    id: string;
    type: NodeType;
    moduleName?: string; // Store the module name for editing
    params: Record<string, any>;
}

interface GraphState {
    nodes: FlowNode[];
    edges: FlowEdge[];
    selectedId: string | null;
    compilerGraph: CompilerGraph;
    jupyter: {
        service: JupyterService | null;
        status: JupyterStatus | null;
        isConnecting: boolean;
        isExecuting: boolean;
        lastExecutionResult: CodeExecutionResult | null;
        config: JupyterConfig | null;
    };
}

interface GraphActions {
    addNode: (config: NodeConfig) => void;
    deleteNode: (id: string) => void;
    setSelectedId: (id: string | null) => void;
    updateNodes: (nodes: FlowNode[]) => void;
    updateEdges: (edges: FlowEdge[]) => void;
    addEdge: (edge: FlowEdge, sourceHandleIndex?: number, targetHandleIndex?: number) => void;
    updateNodeParams: (id: string, params: Record<string, any>) => void;
    connectToJupyter: (config: JupyterConfig) => Promise<JupyterStatus>;
    disconnectFromJupyter: () => Promise<void>;
    executeCodeInJupyter: (code: string, kernelName?: string) => Promise<CodeExecutionResult>;
}

export const useStore = create<GraphState & GraphActions>((set, get) => {
    // Helper to set node error
    const setNodeError = (nodeId: string, error: { input?: string, output?: string }) => 
        set(state => ({
            nodes: state.nodes.map(node => 
                node.id === nodeId
                    ? {
                        ...node,
                        data: {
                            ...node.data,
                            inputError: error.input,
                            outputError: error.output
                        }
                    }
                    : node
            )
        }));

    return {
        // State
        nodes: [],
        edges: [],
        selectedId: null,
        compilerGraph: new CompilerGraph(),
        jupyter: {
            service: null,
            status: null,
            isConnecting: false,
            isExecuting: false,
            lastExecutionResult: null,
            config: null
        },

        // Actions
        /* TODO: Pan to include all nodes when a node is added */
        addNode: (config) => {
            // Create a node in CheckerGraph
            get().compilerGraph.addNode(
                config.id,
                config.type,
                config.params
            );

            // Create a visual node in React Flow
            const flowNode: FlowNode = {
                id: config.id,
                type: 'default',
                data: {
                    type: config.type,
                    moduleName: config.moduleName,
                    params: config.params 
                },
                position: { x: get().nodes.length * GRID_SIZE * 15, y: 0 },
                draggable: true,
            };

            set(state => ({ nodes: [...state.nodes, flowNode] }));
        },

        deleteNode: (id) => {
            // Clean up visual elements
            set(state => ({
                nodes: state.nodes.filter(node => node.id !== id),
                edges: state.edges.filter(edge => 
                    edge.source !== id && edge.target !== id
                ),
                selectedId: state.selectedId === id ? null : state.selectedId
            }));

            get().compilerGraph.removeNode(id);
        },

        setSelectedId: (id) => set({ selectedId: id }),

        updateNodes: (nodes) => set({ nodes }),

        updateEdges: (edges) => set({ edges }),

        addEdge: (edge: FlowEdge, sourceHandleIndex = 0, targetHandleIndex = 0) => {
            // Clear any existing errors
            setNodeError(edge.source, {});
            setNodeError(edge.target, {});

            /* now try to connect in the checker graph */
            try {
                get().compilerGraph.connect(
                    edge.source,
                    edge.target,
                    sourceHandleIndex,
                    targetHandleIndex
                );
            } catch(e: any) {
                setNodeError(edge.source, { output: e.message });
                return;
            }

            /* if it succeeded, connect in visual graph */
            set(state => ({ edges: [...state.edges, edge] }));
        },

        /** TODO: This should be much shorter after we have setParams for every concrete type */
        updateNodeParams: (id, params) => {
            try {
                // Get node from graph
                const node = get().compilerGraph.getNode(id);

                // Store connection info for reconnection
                let connections = {
                            sources: [] as {id: string, sourcePortIndex?: number, targetPortIndex?: number}[],
        targets: [] as {id: string, sourcePortIndex?: number, targetPortIndex?: number}[]
                };

                if (node) {
                    // Node is in the main graph
                    // We need to:
                    // 1. Disconnect it
                    // 2. Remove it
                    // 3. Create a new node with updated params
                    // 4. Reconnect it
                    
                    // First, find all connected edges
                    get().edges.forEach(edge => {
                        if (edge.target === id) {
                            connections.sources.push({
                                id: edge.source,
                                sourcePortIndex: edge.sourceHandle ? parseInt(edge.sourceHandle) : undefined,
                                targetPortIndex: edge.targetHandle ? parseInt(edge.targetHandle) : undefined
                            });
                        }
                        if (edge.source === id) {
                            connections.targets.push({
                                id: edge.target,
                                sourcePortIndex: edge.sourceHandle ? parseInt(edge.sourceHandle) : undefined,
                                targetPortIndex: edge.targetHandle ? parseInt(edge.targetHandle) : undefined
                            });
                        }
                    });
                    
                    // Disconnect all connections
                    [...connections.sources, ...connections.targets].forEach(conn => {
                        try {
                            if (conn.id) {
                                const connectedNode = get().compilerGraph.getNode(conn.id);
                                if (connectedNode) {
                                    if (connections.sources.some(s => s.id === conn.id)) {
                                        // Convert handle IDs to integers
                                        const sourcePortIndex = conn.sourcePortIndex !== undefined ? parseInt(String(conn.sourcePortIndex), 10) : 0;
                                        const targetPortIndex = conn.targetPortIndex !== undefined ? parseInt(String(conn.targetPortIndex), 10) : 0;
                                        get().compilerGraph.disconnect(conn.id, id, sourcePortIndex, targetPortIndex);
                                    } else {
                                        // Convert handle IDs to integers
                                        const sourcePortIndex = conn.sourcePortIndex !== undefined ? parseInt(String(conn.sourcePortIndex), 10) : 0;
                                        const targetPortIndex = conn.targetPortIndex !== undefined ? parseInt(String(conn.targetPortIndex), 10) : 0;
                                        get().compilerGraph.disconnect(id, conn.id, sourcePortIndex, targetPortIndex);
                                    }
                                }
                            }
                        } catch (e) {
                            console.warn('Error disconnecting node:', e);
                        }
                    });
                    
                    // Remove from ZophGraph
                    get().compilerGraph.removeNode(id);
                }
                
                // Find node info from React Flow
                const flowNode = get().nodes.find(n => n.id === id);
                if (!flowNode) throw new Error("Node not found in React Flow");

                get().compilerGraph.addNode(
                    id,
                    flowNode.data.type,
                    params
                );

                // Update React Flow node
                set(state => ({
                    nodes: state.nodes.map(n => 
                        n.id === id 
                            ? {...n, data: {...n.data, params: {...(n.data.params || {}), ...params}}}
                            : n
                    )
                }));

                // Try to reconnect
                connections.sources.forEach(conn => {
                    try {
                        // Parse source and target port indices
                        const sourcePortIndex = conn.sourcePortIndex !== undefined ? parseInt(String(conn.sourcePortIndex), 10) : 0;
                        const targetPortIndex = conn.targetPortIndex !== undefined ? parseInt(String(conn.targetPortIndex), 10) : 0;
                        
                        get().compilerGraph.connect(conn.id, id, sourcePortIndex, targetPortIndex);
                    } catch (e) {
                        console.warn('Error reconnecting source:', e);
                    }
                });
                
                connections.targets.forEach(conn => {
                    try {
                        // Parse source and target port indices
                        const sourcePortIndex = conn.sourcePortIndex !== undefined ? parseInt(String(conn.sourcePortIndex), 10) : 0;
                        const targetPortIndex = conn.targetPortIndex !== undefined ? parseInt(String(conn.targetPortIndex), 10) : 0;
                        
                        get().compilerGraph.connect(id, conn.id, sourcePortIndex, targetPortIndex);
                    } catch (e) {
                        console.warn('Error reconnecting target:', e);
                    }
                });
                
                // Clear any existing errors
                setNodeError(id, {});
            } catch (error) {
                console.error('Failed to update params:', error);
                throw error;
            }
        },
        connectToJupyter: async (config) => {
            set(state => ({
                jupyter: {
                    ...state.jupyter,
                    isConnecting: true,
                    status: null
                }
            }));
            
            try {
                const service = new JupyterService();
                const status = await service.connect(config);
                
                set(state => ({
                    jupyter: {
                        ...state.jupyter,
                        service: status.connected ? service : null,
                        status,
                        isConnecting: false,
                        config: status.connected ? config : null
                    }
                }));
                
                return status;
            } catch (error) {
                console.error('Failed to connect to Jupyter:', error);
                
                const errorStatus = { 
                    connected: false, 
                    error: error instanceof Error ? error.message : String(error)
                };
                
                set(state => ({
                    jupyter: {
                        ...state.jupyter,
                        isConnecting: false,
                        status: errorStatus
                    }
                }));
                
                return errorStatus;
            }
        },
        
        disconnectFromJupyter: async () => {
            const { service } = get().jupyter;
            
            if (service) {
                await service.disconnect();
            }
            
            set(state => ({
                jupyter: {
                    ...state.jupyter,
                    service: null,
                    status: null,
                    config: null
                }
            }));
        },
        
        executeCodeInJupyter: async (code, kernelName) => {
            const { service } = get().jupyter;
            
            if (!service) {
                throw new Error('Not connected to Jupyter');
            }
            
            set(state => ({
                jupyter: {
                    ...state.jupyter,
                    isExecuting: true
                }
            }));
            
            try {
                const result = await service.executeCode(code, kernelName);
                
                set(state => ({
                    jupyter: {
                        ...state.jupyter,
                        isExecuting: false,
                        lastExecutionResult: result
                    }
                }));
                
                return result;
            } catch (error) {
                console.error('Failed to execute code:', error);
                
                const result = {
                    success: false,
                    outputs: [],
                    textOutput: error instanceof Error ? error.message : String(error)
                };
                
                set(state => ({
                    jupyter: {
                        ...state.jupyter,
                        isExecuting: false,
                        lastExecutionResult: result
                    }
                }));
                
                return result;
            }
        }
    };
});
