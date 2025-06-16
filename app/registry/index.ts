import { validateParams as validateTorchParams } from '../../DAGCompiler/torch_nn_module_op';
import { ModuleMetadata } from './types';
import { convolutionalModules } from './convolutional';
import { activationModules } from './activation';
import { normalizationModules } from './normalization';
import { linearModules } from './linear';
import { poolingModules } from './pooling';
import { dropoutModules } from './dropout';
import { reshapeModules } from './reshape';
import { mergeModules } from './merge';
import { branchModules } from './branch';

// Re-export types

// Combine all module metadata
export const allModules: Record<string, ModuleMetadata> = {
    // Tensor nodes
    'Tensor': {
        label: 'Tensor',
        description: 'Represents input data or intermediate results',
        category: 'Basic',
        paramFields: {
            shape: {
                label: 'Shape',
                description: 'Dimensions of the tensor (comma-separated)',
                type: 'shape',
                default: [3, 64, 64],
                allowNegativeOne: false
            },
            variableName: {
                label: 'Variable Name',
                description: 'Name of this tensor',
                type: 'string',
                default: "tensor0"
            }
        }
    },
    // Operation nodes
    ...Object.entries(linearModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(convolutionalModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(reshapeModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(activationModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(dropoutModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(poolingModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    ...Object.entries(normalizationModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
    /* Branch and Merge nodes */
    ...branchModules,
    ...mergeModules
};

// Get metadata for any module by key
export function getMeta(key: string): ModuleMetadata {
    if (!allModules[key]) {
        throw new Error(`No metadata found for module key ${key}`);
    }
    return allModules[key];
}

// Validate parameters for an operation
export function validateParams(opType: string, params: Record<string, any>): string | null {
    try {
        validateTorchParams(opType, params);
        return null;
    } catch (e: any) {
        return e.message;
    }
}
