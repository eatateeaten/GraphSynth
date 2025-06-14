import { ModuleMetadata, ParamFieldMetadata } from './types';

// Common parameter templates
const commonParams: Record<string, ParamFieldMetadata> = {
    num_features: {
        label: 'Number of Features',
        description: 'Number of features or channels expected in the input',
        type: 'number'
    },
    eps: {
        label: 'Epsilon',
        description: 'Value added to the denominator for numerical stability',
        type: 'number',
        default: 1e-5
    },
    momentum: {
        label: 'Momentum',
        description: 'Value used for the running_mean and running_var computation',
        type: 'number',
        default: 0.1
    },
    affine: {
        label: 'Affine',
        description: 'Whether to include learnable affine parameters',
        type: 'option',
        options: ['true', 'false'],
        default: 'true'
    },
    track_running_stats: {
        label: 'Track Running Stats',
        description: 'Whether to track the running mean and variance',
        type: 'option',
        options: ['true', 'false'],
        default: 'true'
    }
};

// Factory for batch normalization modules
function createBatchNormModule(dimension: '1D' | '2D' | '3D'): ModuleMetadata {
    let description: string;
  
    switch (dimension) {
    case '1D':
        description = 'Applies Batch Normalization over a 2D or 3D input';
        break;
    case '2D':
        description = 'Applies Batch Normalization over a 4D input (NCHW)';
        break;
    case '3D':
        description = 'Applies Batch Normalization over a 5D input (NCDHW)';
        break;
    }
  
    return {
        label: `Batch Normalization ${dimension}`,
        description,
        category: 'Normalization',
        paramFields: {
            num_features: commonParams.num_features,
            eps: commonParams.eps,
            momentum: commonParams.momentum,
            affine: commonParams.affine,
            track_running_stats: commonParams.track_running_stats
        }
    };
}

// Factory for instance normalization modules
function createInstanceNormModule(dimension: '1D' | '2D' | '3D'): ModuleMetadata {
    let inputShape: string;

    switch (dimension) {
    case '1D':
        inputShape = '(N, C, L)';
        break;
    case '2D':
        inputShape = '(N, C, H, W)';
        break;
    case '3D':
        inputShape = '(N, C, D, H, W)';
        break;
    }

    return {
        label: `Instance Normalization ${dimension}`,
        description: `Applies Instance Normalization over a ${dimension === '1D' ? '3D' : dimension === '2D' ? '4D' : '5D'} input ${inputShape}`,
        category: 'Normalization',
        paramFields: {
            num_features: commonParams.num_features,
            eps: commonParams.eps,
            momentum: commonParams.momentum,
            affine: {
                ...commonParams.affine,
                default: 'false'
            },
            track_running_stats: {
                ...commonParams.track_running_stats,
                default: 'false'
            }
        }
    };
}

// Generate all normalization module definitions
export const normalizationModules: Record<string, ModuleMetadata> = {
    // Batch Normalization modules
    'BatchNorm1D': createBatchNormModule('1D'),
    'BatchNorm2D': createBatchNormModule('2D'),
    'BatchNorm3D': createBatchNormModule('3D'),

    // Instance Normalization modules
    'InstanceNorm1D': createInstanceNormModule('1D'),
    'InstanceNorm2D': createInstanceNormModule('2D'),
    'InstanceNorm3D': createInstanceNormModule('3D'),

    // Other normalization layers with custom parameters
    'LayerNorm': {
        label: 'Layer Normalization',
        description: 'Applies Layer Normalization over a mini-batch of inputs',
        category: 'Normalization',
        paramFields: {
            normalized_shape: {
                label: 'Normalized Shape',
                description: 'Shape of the normalized tensor (e.g., [10] for sequences, [10, 20] for images)',
                type: 'shape',
                allowNegativeOne: false
            },
            eps: commonParams.eps,
            elementwise_affine: {
                label: 'Elementwise Affine',
                description: 'Whether to apply learnable per-element affine parameters',
                type: 'option',
                options: ['true', 'false'],
                default: 'true'
            }
        }
    },
    'GroupNorm': {
        label: 'Group Normalization',
        description: 'Applies Group Normalization over a mini-batch of inputs',
        category: 'Normalization',
        paramFields: {
            num_groups: {
                label: 'Number of Groups',
                description: 'Number of groups to separate the channels into',
                type: 'number'
            },
            num_channels: {
                label: 'Number of Channels',
                description: 'Number of channels expected in the input',
                type: 'number'
            },
            eps: commonParams.eps,
            affine: commonParams.affine
        }
    }
};
