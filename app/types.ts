// src/types.ts
import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';

export type NodeType = 'tensor' | 'op' | 'merge' | 'branch';

export interface FlowNode extends ReactFlowNode {
    data: {
        type: NodeType;
        opType?: string;
        params?: Record<string, any>;
        inputError?: string;
        outputError?: string;
    };
}

export type FlowEdge = ReactFlowEdge;
