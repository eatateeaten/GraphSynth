import { ModuleDef } from './types';

export const dropoutModules: Record<string, ModuleDef> = {
    'Dropout': {
        label: 'Dropout',
        description: 'Randomly zeroes some elements of the input tensor with probability p during training',
        category: 'Dropout',
        moduleType: 'Op',
        params: {
            p: {
                label: 'Probability',
                description: 'Probability of an element to be zeroed',
                type: 'number',
                default: 0.5,
                required: false
            },
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to apply dropout (optional)',
                type: 'number',
                required: false
            }
        },
        toPytorchModule: (params) => {
            if (params.dim !== undefined) {
                return `nn.functional.dropout(input, p=${params.p ?? 0.5}, training=self.training, inplace=False, dim=${params.dim})`;
            }
            return `nn.Dropout(p=${params.p ?? 0.5}, inplace=False)`;
        },
        validateInputShape: (inShape, params) => {
            if (inShape.length === 0) {
                return [`Dropout requires at least 1D input tensor, got shape ${inShape}`];
            }
            if (params.dim !== undefined) {
                const dim = params.dim;
                if (dim < 0 || dim >= inShape.length) {
                    return [`Dropout dimension ${dim} is out of bounds for input shape ${inShape}`];
                }
            }
            return [];
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape];
        }
    },

    'Dropout2D': {
        label: 'Dropout2D',
        description: 'Randomly zero out entire channels (a channel is a 2D feature map)',
        category: 'Dropout',
        moduleType: 'Op',
        params: {
            p: {
                label: 'Probability',
                description: 'Probability of an element to be zeroed',
                type: 'number',
                default: 0.5,
                required: false
            },
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to apply dropout (optional)',
                type: 'number',
                required: false
            }
        },
        toPytorchModule: (params) => {
            if (params.dim !== undefined) {
                return `nn.functional.dropout2d(input, p=${params.p ?? 0.5}, training=self.training, inplace=False, dim=${params.dim})`;
            }
            return `nn.Dropout2d(p=${params.p ?? 0.5}, inplace=False)`;
        },
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4) {
                return [`Dropout2D typically expects 4D input tensor, got shape ${inShape}`];
            }
            if (params.dim !== undefined) {
                const dim = params.dim;
                if (dim < 0 || dim >= inShape.length) {
                    return [`Dropout2D dimension ${dim} is out of bounds for input shape ${inShape}`];
                }
            }
            return [];
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape];
        }
    },

    'Dropout3D': {
        label: 'Dropout3D',
        description: 'Randomly zero out entire channels (a channel is a 3D feature map)',
        category: 'Dropout',
        moduleType: 'Op',
        params: {
            p: {
                label: 'Probability',
                description: 'Probability of an element to be zeroed',
                type: 'number',
                default: 0.5,
                required: false
            },
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to apply dropout (optional)',
                type: 'number',
                required: false
            }
        },
        toPytorchModule: (params) => {
            if (params.dim !== undefined) {
                return `nn.functional.dropout3d(input, p=${params.p ?? 0.5}, training=self.training, inplace=False, dim=${params.dim})`;
            }
            return `nn.Dropout3d(p=${params.p ?? 0.5}, inplace=False)`;
        },
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 5) {
                return [`Dropout3D typically expects 5D input tensor, got shape ${inShape}`];
            }
            if (params.dim !== undefined) {
                const dim = params.dim;
                if (dim < 0 || dim >= inShape.length) {
                    return [`Dropout3D dimension ${dim} is out of bounds for input shape ${inShape}`];
                }
            }
            return [];
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape];
        }
    },

    'AlphaDropout': {
        label: 'Alpha Dropout',
        description: 'Applies Alpha Dropout over the input (maintains the self-normalizing property in conjunction with SELU)',
        category: 'Dropout',
        moduleType: 'Op',
        params: {
            p: {
                label: 'Probability',
                description: 'Probability of an element to be zeroed',
                type: 'number',
                default: 0.5,
                required: false
            },
            dim: {
                label: 'Dimension',
                description: 'Dimension along which to apply dropout (optional)',
                type: 'number',
                required: false
            }
        },
        toPytorchModule: (params) => {
            if (params.dim !== undefined) {
                return `nn.functional.alpha_dropout(input, p=${params.p ?? 0.5}, training=self.training, inplace=False, dim=${params.dim})`;
            }
            return `nn.AlphaDropout(p=${params.p ?? 0.5}, inplace=False)`;
        },
        validateInputShape: (inShape, params) => {
            if (inShape.length === 0) {
                return [`AlphaDropout requires at least 1D input tensor, got shape ${inShape}`];
            }
            if (params.dim !== undefined) {
                const dim = params.dim;
                if (dim < 0 || dim >= inShape.length) {
                    return [`AlphaDropout dimension ${dim} is out of bounds for input shape ${inShape}`];
                }
            }
            return [];
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape];
        }
    }
};
