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
    validateInputShape: (inShape: number[], params: Record<string, any>) => string[];
    inferOutputShape: (inShape: number[], params: Record<string, any>) => number[];
}
