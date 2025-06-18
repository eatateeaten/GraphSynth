import { NodeType } from "../OpCompiler/types";

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
    moduleType: NodeType;

    // Parameter definitions
    params: Record<string, ParamDef>;

    // Computational logic
    toPytorchModule: (params: Record<string, any>) => string;
    // Required for Op nodes (SISO), optional for multi-input operations handled by OpCompiler
    validateInputShape?: ((inShapes: number[], params: Record<string, any>) => string[]) | null;
    inferOutputShape?: ((inShapes: number[], params: Record<string, any>) => number[]) | null;
}
