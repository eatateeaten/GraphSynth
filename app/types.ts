// src/types.ts
import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';

export type NodeType = 'Tensor' | 'Op' | 'Split' | 'Copy' | 'Concat' | 'PointwiseReduce' | 'DotOp' | 'CrossOp';

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
