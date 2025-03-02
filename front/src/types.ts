import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';
import { CheckerNode, Tensor, Reshape } from './CheckerGraph';

export type LayerType = 'tensor' | 'reshape';

// Just the minimal info needed to create a node
export interface LayerConfig {
  type: LayerType;
  params: Record<string, any>;
}

// Only store UI-specific data in the Flow node
export interface FlowNode extends ReactFlowNode {
  data: {
    type: LayerType;
    errorMessage?: string;
  };
}

export type FlowEdge = ReactFlowEdge;

// Factory function to create CheckerNode instances
export function createCheckerNode(config: LayerConfig): CheckerNode {
  switch (config.type) {
    case 'tensor':
      return new Tensor(config.params);
    case 'reshape':
      return new Reshape(config.params);
    default:
      throw new Error(`Unknown layer type: ${config.type}`);
  }
}
