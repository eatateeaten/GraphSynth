/**
 * This store implements a global state management pattern using Zustand.
 * Rather than prop-drilling or managing distributed state across components,
 * we centralize our application state here for predictable data flow and updates.
 * 
 * The store maintains two parallel graph representations:
 * 1. CheckerGraph: Domain logic layer
 *    - Validates tensor shapes and parameter constraints
 *    - Manages node connectivity and data flow
 *    - Source of truth for computational graph structure
 * 
 * 2. React Flow: Presentation layer
 *    - Handles node positioning and visual layout
 *    - Manages UI state (selection, dragging, etc)
 *    - Displays validation errors from CheckerGraph
 * 
 * Operations follow a consistent pattern:
 * 1. Validate operation against CheckerGraph (shape checking, valid connections)
 * 2. On success: update React Flow state
 * 3. On failure: propagate error to UI layer
 */

import { create } from 'zustand';
import type { FlowNode, FlowEdge } from './types';
import { createCheckerNode, CheckerNodeConfig, CheckerGraph } from './checker';
import { GRID_SIZE } from './config';

interface GraphState {
  nodes: FlowNode[];
  edges: FlowEdge[];
  selectedId: string | null;
  checkerGraph: CheckerGraph;
}

interface GraphActions {
  addNode: (id: string, config: CheckerNodeConfig) => void;
  deleteNode: (id: string) => void;
  setSelectedId: (id: string | null) => void;
  updateNodes: (nodes: FlowNode[]) => void;
  updateEdges: (edges: FlowEdge[]) => void;
  addEdge: (edge: FlowEdge) => void;
  updateNodeParams: (id: string, params: Record<string, any>) => void;
}

export const useGraphStore = create<GraphState & GraphActions>((set, get) => {
  // Helper to set error message on nodes
  const setNodeError = (ids: string[], error?: string) => 
    set(state => ({
      nodes: state.nodes.map(node => 
        ids.includes(node.id)
          ? { ...node, data: { ...node.data, errorMessage: error } }
          : node
      )
    }));

  return {
    // State
    nodes: [],
    edges: [],
    selectedId: null,
    checkerGraph: new CheckerGraph(),

    // Actions
    addNode: (id, config) => {
      try {
        const checkerNode = createCheckerNode(config);
        get().checkerGraph.addNode(id, checkerNode);

        const flowNode: FlowNode = {
          id,
          type: 'default',
          data: { type: config.type },
          position: { x: get().nodes.length * GRID_SIZE * 15, y: 0 },
          draggable: true,
        };

        set(state => ({ nodes: [...state.nodes, flowNode] }));
      } catch (e) {
        const flowNode: FlowNode = {
          id,
          type: 'default',
          data: { 
            type: config.type,
            errorMessage: e instanceof Error ? e.message : 'Unknown error'
          },
          position: { x: get().nodes.length * GRID_SIZE * 15, y: 0 },
          draggable: true,
        };
        set(state => ({ nodes: [...state.nodes, flowNode] }));
      }
    },

    deleteNode: (id) => {
      get().checkerGraph.deleteNode(id);
      set(state => ({
        nodes: state.nodes.filter(node => node.id !== id),
        edges: state.edges.filter(edge => 
          edge.source !== id && edge.target !== id
        ),
        selectedId: null
      }));
    },

    setSelectedId: (id) => set({ selectedId: id }),

    updateNodes: (nodes) => set({ nodes }),

    updateEdges: (edges) => set({ edges }),

    addEdge: (edge) => {
      try {
        get().checkerGraph.connect(edge.source, edge.target);
        set(state => ({ edges: [...state.edges, edge] }));
        setNodeError([edge.source, edge.target]);
      } catch (e) {
        setNodeError(
          [edge.source, edge.target],
          e instanceof Error ? e.message : 'Unknown error'
        );
      }
    },

    updateNodeParams: (id, params) => {
      try {
        const node = get().checkerGraph.getNode(id);
        if (!node) throw new Error("Node not found");
        node.set_params(params);
        setNodeError([id]);
      } catch (e) {
        setNodeError([id], e instanceof Error ? e.message : 'Unknown error');
      }
    }
  };
});
