import type { GraphNode } from './graph_node';

/** Factory function type for creating nodes */
export type NodeFactory = (id: string, params: Record<string, any>) => GraphNode;

/** Registry of node type names to their factory functions */
export class NodeTypeRegistry {
    private static _registry = new Map<string, NodeFactory>();
    
    static register(typeName: string, factory: NodeFactory): void {
        this._registry.set(typeName, factory);
    }
    
    static get(typeName: string): NodeFactory | undefined {
        return this._registry.get(typeName);
    }
    
    static has(typeName: string): boolean {
        return this._registry.has(typeName);
    }
    
    static getAllTypes(): string[] {
        return Array.from(this._registry.keys());
    }
}

/** Node type enum that will be populated dynamically */
export enum NodeType {
    // Basic types
    TENSOR = 'Tensor',
    OP = 'Op',
    
    // Branch operations
    SPLIT = 'Split',
    COPY = 'Copy',
    
    // Merge operations  
    POINTWISE_OP = 'PointwiseOp',
    DOT_OP = 'DotOp',
    CROSS_OP = 'CrossOp',
    
    // Reduce operations
    CONCAT = 'Concat',
    POINTWISE_REDUCE = 'PointwiseReduce'
}

export function isNodeType(str: string): str is NodeType {
    return NodeTypeRegistry.has(str) || Object.values(NodeType).includes(str as NodeType);
}

export type Shape = number[];



