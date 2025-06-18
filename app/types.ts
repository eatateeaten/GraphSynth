// src/types.ts
import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';
import type { NodeType } from "../OpCompiler/types";

export interface FlowNode extends ReactFlowNode {
    data: {
        type: NodeType;
        moduleName?: string;
        params?: Record<string, any>;
        inputError?: string;
        outputError?: string;
    };
}

export type FlowEdge = ReactFlowEdge;
