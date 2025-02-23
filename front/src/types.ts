import type { Node as FlowNode } from 'reactflow';

export type Sourceness = 'source' | 'middle' | 'sink';  // How the node participates in data flow

export type LayerType = 
  | 'tensor' 
  | 'reshape';  // Only keeping types that exist in node.py

export interface LayerParams {
  data?: number[];      // For tensor data
  shape?: number[];     // For tensor shape
  out_dim?: number[];   // For reshape dimensions
}

export interface Layer {
  id: string;           // Frontend ID (for React Flow)
  serverId?: string;    // Backend ID (from server)
  name: string;
  type: LayerType;
  sourceness: Sourceness;  // Whether node generates, transforms, or consumes data
  params: LayerParams;
  inShape?: number[];
  outShape?: number[];
  isValid?: boolean;
}

export type Sequence = {
  name: string;
  nodes: Layer[];
};

// WebSocket Messages
export interface WSRequest {
  requestId: string;
  operation: 'addNode' | 'setInputNode' | 'setOutputNode';
  nodeId?: string;      // Use server ID
  inputId?: string;     // Use server ID
  outputId?: string;    // Use server ID
  type?: LayerType;
  params?: LayerParams;
}

export interface WSResponse {
  success: boolean;
  requestId?: string;
  id?: string;          // Server ID
  error?: string;
  in_shape?: number[];
  out_shape?: number[];
  completed?: boolean;
}

// Node UI States
export interface NodeData extends Layer {
  status: 'bubble' | 'validating' | 'valid' | 'error';
  errorMessage?: string;
}

export interface GraphNode extends FlowNode {
  data: NodeData;
}

// Helper to convert snake_case to Title Case
export function formatLabel(value: string): string {
  return value
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join('');
}
