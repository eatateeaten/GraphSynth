import { ModuleMetadata } from './types';

export const reshapeModules: Record<string, ModuleMetadata> = {
    'Reshape': {
        label: 'Reshape',
        description: 'Reshapes the input tensor to the specified shape',
        category: 'Reshape',
        paramFields: {
            shape: {
                label: 'Shape',
                description: 'The desired shape (comma-separated numbers)',
                type: 'shape',
                allowNegativeOne: true
            }
        }
    },
    'Permute': {
        label: 'Permute',
        description: 'Permutes the dimensions of the input tensor',
        category: 'Reshape',
        paramFields: {
            dims: {
                label: 'Dimensions',
                description: 'The desired ordering of dimensions (comma-separated numbers)',
                type: 'shape',
                allowNegativeOne: false
            }
        }
    },
    'Flatten': {
        label: 'Flatten',
        description: 'Flattens input by reshaping it into a one-dimensional tensor',
        category: 'Reshape',
        paramFields: {
            start_dim: {
                label: 'Start Dimension',
                description: 'First dim to flatten',
                type: 'number',
                default: 1
            },
            end_dim: {
                label: 'End Dimension',
                description: 'Last dim to flatten',
                type: 'number',
                default: -1
            }
        }
    },
    'Unflatten': {
        label: 'Unflatten',
        description: 'Unflattens a tensor dim expanding it to a desired shape',
        category: 'Reshape',
        paramFields: {
            dim: {
                label: 'Dimension',
                description: 'Dimension to unflatten',
                type: 'number',
                default: 1
            },
            unflattened_size: {
                label: 'Unflattened Size',
                description: 'New shape of the unflattened dimension',
                type: 'shape',
                allowNegativeOne: false
            }
        }
    }
};
