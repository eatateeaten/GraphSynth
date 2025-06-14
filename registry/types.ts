export type { NodeType } from '../app/types';

export interface ParamFieldMetadata {
    label: string;
    description: string;
    type: 'number' | 'shape' | 'option' | 'boolean' | 'string';
    default?: any;
    options?: string[];
    allowNegativeOne?: boolean;
    placeholder?: string;
}

export interface ModuleMetadata {
    label: string;
    description: string;
    category: string;
    paramFields: Record<string, ParamFieldMetadata>;
}

export type Shape = number[];
