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
    }
};
