export type { NodeType } from '../types';

export interface ParamFieldMetadata {
    label: string;
    description: string;
    type: 'number' | 'shape' | 'option';
    default?: any;
    options?: string[];
    allowNegativeOne?: boolean;
}

export interface ModuleMetadata {
    label: string;
    description: string;
    category: string;
    paramFields: Record<string, ParamFieldMetadata>;
}

export type Shape = number[];
