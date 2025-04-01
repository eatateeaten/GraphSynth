import { ModuleMetadata } from './types';

export const mergeModules: Record<string, ModuleMetadata> = {
    'Concat': {
        label: 'Concat',
        description: 'Concatenates multiple tensors along a specified dimension',
        category: 'Flow',
        paramFields: {
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to concatenate the tensors',
                type: 'number',
                default: 0
            }
        }
    }
};
