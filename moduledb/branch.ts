import { ModuleDef } from './types';

export const Split: ModuleDef = {
    label: 'Split',
    description: 'Splits a tensor into multiple tensors along a specified dimension',
    category: 'Flow',
    moduleType: 'Split',
    params: {
        dim: {
            label: 'Dimension',
            description: 'Dimension along which to split the tensor',
            type: 'number',
            default: 0,
            required: false
        },
        sections: {
            label: 'Sections',
            description: 'Sizes of each section (comma-separated)',
            type: 'shape',
            default: [1, 1],
            allowNegativeOne: false,
            required: false
        }
    },
    toPytorchModule: (params: Record<string, any>) => {
        const dim = params.dim ?? 0;
        const sections = params.sections ?? [1, 1];
        return `torch.split(x, ${JSON.stringify(sections)}, dim=${dim})`;
    },
    validateInputShape: null,
    inferOutputShape: null
};

export const Copy: ModuleDef = {
    label: 'Copy',
    description: 'Creates multiple identical copies of the input tensor',
    category: 'Flow',
    moduleType: 'Copy',
    params: {
        copies: {
            label: 'Number of Copies',
            description: 'Number of identical copies to create',
            type: 'number',
            default: 2,
            required: false
        }
    },
    /* TODO: probably doesn't work */
    toPytorchModule: (params: Record<string, any>) => {
        const copies = params.copies ?? 2;
        return `tuple(x for _ in range(${copies}))`;
    },
    validateInputShape: null,
    inferOutputShape: null
};
