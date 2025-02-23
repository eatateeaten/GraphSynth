import { create } from 'zustand';
import type { Edge, Connection } from 'reactflow';
import type { Layer, GraphNode, NodeData } from './types';
import { GRID_SIZE } from './config';

interface GraphState {
  layers: Layer[];
  nodes: GraphNode[];
  edges: Edge[];
  selectedNode: string | null;
}

interface GraphActions {
  addLayer: (layer: Layer) => void;
  deleteNode: (nodeId: string) => void;
  setSelectedNode: (nodeId: string | null) => void;
  updateNodes: (nodes: GraphNode[]) => void;
  updateEdges: (edges: Edge[]) => void;
  addEdge: (edge: Edge) => void;
  updateNodeData: (nodeId: string, data: Partial<NodeData> & { serverId?: string }) => void;
}

export const useGraphStore = create<GraphState & GraphActions>((set, get) => ({
  // State
  layers: [],
  nodes: [],
  edges: [],
  selectedNode: null,

  // Actions
  addLayer: (layer) => {
    // Create node in bubble state
    const nodeData: NodeData = {
      ...layer,
      status: 'bubble',
      inShape: undefined,
      outShape: undefined
    };

    const flowNode: GraphNode = {
      id: layer.id,
      type: 'default',
      data: nodeData,
      position: { x: get().nodes.length * GRID_SIZE * 15, y: 0 },
      draggable: true,
    };

    set(state => ({
      layers: [...state.layers, layer],
      nodes: [...state.nodes, flowNode]
    }));
  },

  deleteNode: (nodeId) => set(state => ({
    nodes: state.nodes.filter(node => node.id !== nodeId),
    edges: state.edges.filter(edge => 
      edge.source !== nodeId && edge.target !== nodeId
    ),
    layers: state.layers.filter(layer => layer.id !== nodeId),
    selectedNode: null
  })),

  setSelectedNode: (nodeId) => set({ selectedNode: nodeId }),

  updateNodes: (nodes) => set({ nodes }),

  updateEdges: (edges) => set({ edges }),

  addEdge: (edge) => set(state => ({
    edges: [...state.edges, edge]
  })),

  updateNodeData: (nodeId, data) => set(state => {
    // Update node data including server ID if provided
    const updatedNodes = state.nodes.map(node => 
      node.id === nodeId
        ? { 
            ...node,
            data: { 
              ...node.data,
              ...data,
            }
          }
        : node
    );

    // Update layers with the same data
    const updatedLayers = state.layers.map(layer =>
      layer.id === nodeId
        ? { 
            ...layer,
            ...data,
          }
        : layer
    );

    return {
      nodes: updatedNodes,
      layers: updatedLayers
    };
  })
}));
