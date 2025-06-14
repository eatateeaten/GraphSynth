import { ModuleMetadata } from './types';

export const branchModules: Record<string, ModuleMetadata> = {
    'Split': {
        label: 'Split',
        description: 'Splits a tensor into multiple tensors along a specified dimension',
        category: 'Flow',
        paramFields: {
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to split the tensor',
                type: 'number',
                default: 0
            },
            sections: {
                label: 'Sections',
                description: 'Sizes of each section (comma-separated)',
                type: 'shape',
                default: [1, 1],
                allowNegativeOne: false
            }
        }
    },
    'Copy': {
        label: 'Copy',
        description: 'Creates multiple identical copies of the input tensor',
        category: 'Flow',
        paramFields: {
            copies: {
                label: 'Number of Copies',
                description: 'Number of identical copies to create',
                type: 'number',
                default: 2
            }
        }
    }
};
