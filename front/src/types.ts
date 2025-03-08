// src/types.ts
import type { Node as ReactFlowNode, Edge as ReactFlowEdge } from 'reactflow';
import { CheckerNodeType } from './checker';

export interface FlowNode extends ReactFlowNode {
    data: {
        type: CheckerNodeType;
        inputError?: string;
        outputError?: string;
    };
}

export type FlowEdge = ReactFlowEdge;

export { createCheckerNode } from './checker';
