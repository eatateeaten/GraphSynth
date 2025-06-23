import { NodeType } from "../OpCompiler/types";
import type { GraphNode } from "../OpCompiler/graph_node";

export interface ParamDef {
    // UI metadata
    label: string;
    description: string;
    type: 'number' | 'shape' | 'option' | 'boolean' | 'string';
    default?: any;
    options?: string[];
    allowNegativeOne?: boolean;
    placeholder?: string;
    
    // Computational metadata
    required: boolean;
}

export interface ModuleDef {
    // UI metadata
    label: string;
    description: string;
    category: string;
    
    // Dynamic node system
    nodeClass?: new (id: string, params: Record<string, any>) => GraphNode;
    hierarchy?: string[];
    tags?: string[];
    
    // Legacy support - will be deprecated
    moduleType?: NodeType;

    // Parameter definitions
    params: Record<string, ParamDef>;

    // Computational logic
    emitPytorchModule?: (params: Record<string, any>) => string;
    // Required for Op nodes (SISO), optional for multi-input operations handled by OpCompiler
    validateInputShape?: ((inShapes: number[], params: Record<string, any>) => string[]) | null;
    inferOutputShape?: ((inShapes: number[], params: Record<string, any>) => number[]) | null;
}
