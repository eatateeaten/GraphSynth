/**
 * This store implements a global state management pattern using Zustand.
 * Rather than prop-drilling or managing distributed state across components,
 * we centralize our application state here for predictable data flow and updates.
 * 
 * The store maintains two parallel graph representations:
 * 1. ZophGraph: Domain logic layer
 *        - Validates tensor shapes and parameter constraints
 *        - Manages node connectivity and data flow
 *        - Source of truth for computational graph structure
 * 
 * 2. React Flow: Presentation layer
 *        - Handles node positioning and visual layout
 *        - Manages UI state (selection, dragging, etc)
 *        - Displays validation errors from ZophGraph
 * 
 * Operations follow a consistent pattern:
 * 1. Validate operation against ZophGraph (shape checking, valid connections)
 * 2. On success: update React Flow state
 * 3. On failure: propagate error to UI layer
 */

import { create } from 'zustand';
import type { FlowNode, FlowEdge  } from './types';
import { GRID_SIZE } from './config';

import { Graph as CheckerGraph } from '../isomorphic/graph';

interface NodeConfig {
    id: string;
    type: 'tensor' | 'op' | 'merge' | 'branch';
    opType?: string; // For operations, e.g., "Conv2d", "ReLU"
    params: Record<string, any>;
}

interface GraphState {
    nodes: FlowNode[];
    edges: FlowEdge[];
    selectedId: string | null;
    checkerGraph: CheckerGraph;
    // Track pending nodes that aren't in the main graph yet
    pendingNodeIds: Set<string>;
}

interface GraphActions {
    addNode: (config: NodeConfig) => void;
    deleteNode: (id: string) => void;
    setSelectedId: (id: string | null) => void;
    updateNodes: (nodes: FlowNode[]) => void;
    updateEdges: (edges: FlowEdge[]) => void;
    addEdge: (edge: FlowEdge) => void;
    updateNodeParams: (id: string, params: Record<string, any>) => void;
}

export const useGraphStore = create<GraphState & GraphActions>((set, get) => {
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
        checkerGraph: new CheckerGraph(),
        pendingNodeIds: new Set<string>(),

        // Actions
        addNode: (config) => {
            try {
                // Create a pending node in ZophGraph
                get().checkerGraph.createPendingNode(
                    config.type === 'tensor' ? 'Tensor' : 
                    config.type === 'op' ? 'Op' : config.type,
                    config.id,
                    {
                        target: "torch",
                        ...config.params,
                        opType: config.opType,
                    }
                );
                
                // Track this ID as pending
                set(state => ({
                    pendingNodeIds: new Set([...state.pendingNodeIds, config.id])
                }));

                // Create a visual node in React Flow
                const flowNode: FlowNode = {
                    id: config.id,
                    type: 'default',
                    data: { 
                        type: config.type,
                        opType: config.opType,
                        params: config.params 
                    },
                    position: { x: get().nodes.length * GRID_SIZE * 15, y: 0 },
                    draggable: true,
                };

                set(state => ({ nodes: [...state.nodes, flowNode] }));
            } catch (e) {
                // Don't create visual node if node creation failed
                console.error('Failed to create node:', e);
                throw e;
            }
        },

        deleteNode: (id) => {
            try {
                // Check if node is in main graph
                const node = get().checkerGraph.getNode(id);
                if (node) {
                    // If it's in the main graph, remove it
                    get().checkerGraph.removeNode(id);
                } else if (get().pendingNodeIds.has(id)) {
                    // If it's a pending node, remove it from pending
                    get().checkerGraph.removePendingNode(id);
                    
                    // Update our pending nodes set
                    const pendingNodeIds = new Set(get().pendingNodeIds);
                    pendingNodeIds.delete(id);
                    set({ pendingNodeIds });
                }
                
                // Clean up visual elements
                set(state => ({
                    nodes: state.nodes.filter(node => node.id !== id),
                    edges: state.edges.filter(edge => 
                        edge.source !== id && edge.target !== id
                    ),
                    selectedId: state.selectedId === id ? null : state.selectedId
                }));
            } catch (e) {
                console.error('Failed to delete node:', e);
                throw e;
            }
        },

        setSelectedId: (id) => set({ selectedId: id }),

        updateNodes: (nodes) => set({ nodes }),

        updateEdges: (edges) => set({ edges }),

        addEdge: (edge) => {
            // Clear any existing errors
            setNodeError(edge.source, {});
            setNodeError(edge.target, {});

            try {
                // Connect nodes in ZophGraph
                get().checkerGraph.connect(
                    edge.source, 
                    edge.target,
                    // Convert string handles to numbers if provided
                    edge.sourceHandle ? parseInt(edge.sourceHandle) : undefined,
                    edge.targetHandle ? parseInt(edge.targetHandle) : undefined
                );
                
                // Check if connection moved nodes from pending to main graph
                const sourceNode = get().checkerGraph.getNode(edge.source);
                const targetNode = get().checkerGraph.getNode(edge.target);
                
                // Update our pending nodes tracking
                if (sourceNode && get().pendingNodeIds.has(edge.source)) {
                    const pendingNodeIds = new Set(get().pendingNodeIds);
                    pendingNodeIds.delete(edge.source);
                    set({ pendingNodeIds });
                }
                
                if (targetNode && get().pendingNodeIds.has(edge.target)) {
                    const pendingNodeIds = new Set(get().pendingNodeIds);
                    pendingNodeIds.delete(edge.target);
                    set({ pendingNodeIds });
                }
                
                // Add visual edge if connection succeeded
                set(state => ({ edges: [...state.edges, edge] }));
            } catch (error) {
                console.error('Connection error:', error);
                
                // Handle error based on message contents
                if (error instanceof Error) {
                    const errorMsg = error.message.toLowerCase();
                    
                    if (errorMsg.includes('shape mismatch') || 
                        errorMsg.includes('output shape') || 
                        errorMsg.includes('cannot connect from')) {
                        // Error from source node's output
                        setNodeError(edge.source, { output: error.message });
                    } else if (errorMsg.includes('input shape') || 
                               errorMsg.includes('cannot connect to')) {
                        // Error from target node's input
                        setNodeError(edge.target, { input: error.message });
                    } else {
                        console.error('Failed to add edge:', error);
                        throw error; // Let unknown errors propagate
                    }
                } else {
                    console.error('Failed to add edge (unknown error type):', error);
                    throw error;
                }
            }
        },

        updateNodeParams: (id, params) => {
            try {
                // Get node from graph
                const node = get().checkerGraph.getNode(id);
                
                // Store connection info for reconnection
                let connections = {
                    sources: [] as {id: string, sourceIndex?: number, targetIndex?: number}[],
                    targets: [] as {id: string, sourceIndex?: number, targetIndex?: number}[]
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
                                sourceIndex: edge.sourceHandle ? parseInt(edge.sourceHandle) : undefined,
                                targetIndex: edge.targetHandle ? parseInt(edge.targetHandle) : undefined
                            });
                        }
                        if (edge.source === id) {
                            connections.targets.push({
                                id: edge.target,
                                sourceIndex: edge.sourceHandle ? parseInt(edge.sourceHandle) : undefined,
                                targetIndex: edge.targetHandle ? parseInt(edge.targetHandle) : undefined
                            });
                        }
                    });
                    
                    // Disconnect all connections
                    [...connections.sources, ...connections.targets].forEach(conn => {
                        try {
                            if (conn.id) {
                                const connectedNode = get().checkerGraph.getNode(conn.id);
                                if (connectedNode) {
                                    if (connections.sources.some(s => s.id === conn.id)) {
                                        get().checkerGraph.disconnect(conn.id, id, conn.sourceIndex, conn.targetIndex);
                                    } else {
                                        get().checkerGraph.disconnect(id, conn.id, conn.sourceIndex, conn.targetIndex);
                                    }
                                }
                            }
                        } catch (e) {
                            console.warn('Error disconnecting node:', e);
                        }
                    });
                    
                    // Remove from ZophGraph
                    get().checkerGraph.removeNode(id);
                }
                
                // Track whether node was pending
                const wasPending = get().pendingNodeIds.has(id);
                
                // Find node info from React Flow
                const flowNode = get().nodes.find(n => n.id === id);
                if (!flowNode) throw new Error("Node not found in React Flow");
                
                // Create a new node with updated params
                const nodeType = flowNode.data.type === 'tensor' ? 'Tensor' : 
                              flowNode.data.type === 'op' ? 'Op' : flowNode.data.type;
                
                // Create a pending node
                get().checkerGraph.createPendingNode(
                    nodeType,
                    id,
                    {
                        target: "torch",
                        ...(flowNode.data.params || {}),
                        ...params,
                        opType: flowNode.data.opType
                    }
                );
                
                // Mark as pending if it wasn't already
                if (!wasPending) {
                    set(state => ({
                        pendingNodeIds: new Set([...state.pendingNodeIds, id])
                    }));
                }
                
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
                        get().checkerGraph.connect(conn.id, id, conn.sourceIndex, conn.targetIndex);
                    } catch (e) {
                        console.warn('Error reconnecting source:', e);
                    }
                });
                
                connections.targets.forEach(conn => {
                    try {
                        get().checkerGraph.connect(id, conn.id, conn.sourceIndex, conn.targetIndex);
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
        }
    };
});
