import { ModuleDef, ParamDef } from './types';

// Shared parameter definitions for normalization modules
const commonParams: Record<string, ParamDef> = {
    num_features: {
        label: 'Number of Features',
        description: 'Number of features or channels expected in the input',
        type: 'number',
        required: true
    },
    eps: {
        label: 'Epsilon',
        description: 'Value added to the denominator for numerical stability',
        type: 'number',
        default: 1e-5,
        required: false
    },
    momentum: {
        label: 'Momentum',
        description: 'Value used for the running_mean and running_var computation',
        type: 'number',
        default: 0.1,
        required: false
    },
    affine: {
        label: 'Affine',
        description: 'Whether to include learnable affine parameters',
        type: 'boolean',
        default: true,
        required: false
    },
    track_running_stats: {
        label: 'Track Running Stats',
        description: 'Whether to track the running mean and variance',
        type: 'boolean',
        default: true,
        required: false
    },
    elementwise_affine: {
        label: 'Elementwise Affine',
        description: 'Whether to apply learnable per-element affine parameters',
        type: 'boolean',
        default: true,
        required: false
    },
    normalized_shape: {
        label: 'Normalized Shape',
        description: 'Shape of the normalized tensor (e.g., [10] for sequences, [10, 20] for images)',
        type: 'shape',
        allowNegativeOne: false,
        required: true
    },
    num_groups: {
        label: 'Number of Groups',
        description: 'Number of groups to separate the channels into',
        type: 'number',
        required: true
    },
    num_channels: {
        label: 'Number of Channels',
        description: 'Number of channels expected in the input',
        type: 'number',
        required: true
    }
};

// Factory function for batch normalization modules
function createBatchNormModule(dimension: '1D' | '2D' | '3D'): ModuleDef {
    const dimensionInfo = {
        '1D': { 
            label: 'Batch Normalization 1D',
            description: 'Applies Batch Normalization over a 2D or 3D input',
            funcName: 'BatchNorm1d',
            validateShape: (inShape: number[]) => {
                const errors: string[] = [];
                if (inShape.length !== 2 && inShape.length !== 3) {
                    errors.push(`BatchNorm1D requires 2D or 3D input tensor, got shape ${inShape}`);
                }
                return errors;
            }
        },
        '2D': { 
            label: 'Batch Normalization 2D',
            description: 'Applies Batch Normalization over a 4D input (NCHW)',
            funcName: 'BatchNorm2d',
            validateShape: (inShape: number[], params: Record<string, any>) => {
                const errors: string[] = [];
                const is4D = inShape.length === 4;
                const is3D = inShape.length === 3;
                
                if (!is4D && !is3D) {
                    errors.push(`BatchNorm2D requires 3D or 4D input tensor, got shape ${inShape}`);
                } else {
                    const channelDim = is4D ? 1 : 0;
                    if (inShape[channelDim] !== params['num_features']) {
                        errors.push(`BatchNorm2D expected num_features=${params['num_features']}, got ${inShape[channelDim]}`);
                    }
                }
                return errors;
            }
        },
        '3D': { 
            label: 'Batch Normalization 3D',
            description: 'Applies Batch Normalization over a 5D input (NCDHW)',
            funcName: 'BatchNorm3d',
            validateShape: (inShape: number[], params: Record<string, any>) => {
                const errors: string[] = [];
                const is5D = inShape.length === 5;
                const is4D = inShape.length === 4;
                
                if (!is5D && !is4D) {
                    errors.push(`BatchNorm3D requires 4D or 5D input tensor, got shape ${inShape}`);
                } else {
                    const channelDim = is5D ? 1 : 0;
                    if (inShape[channelDim] !== params['num_features']) {
                        errors.push(`BatchNorm3D expected num_features=${params['num_features']}, got ${inShape[channelDim]}`);
                    }
                }
                return errors;
            }
        }
    };

    const info = dimensionInfo[dimension];
    
    return {
        label: info.label,
        description: info.description,
        category: 'Normalization',
        moduleType: 'Op',
        params: {
            num_features: commonParams.num_features,
            eps: commonParams.eps,
            momentum: commonParams.momentum,
            affine: commonParams.affine,
            track_running_stats: commonParams.track_running_stats
        },
        toPytorchModule: (params) => {
            return `nn.${info.funcName}(num_features=${params['num_features']}, ` +
                `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
                `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true})`;
        },
        validateInputShape: (inShape, params) => info.validateShape(inShape, params),
        inferOutputShape: (inShape, params) => [...inShape]
    };
}

// Factory function for instance normalization modules
function createInstanceNormModule(dimension: '1D' | '2D' | '3D'): ModuleDef {
    const dimensionInfo = {
        '1D': { 
            label: 'Instance Normalization 1D',
            description: 'Applies Instance Normalization over a 3D input (N, C, L)',
            funcName: 'InstanceNorm1d'
        },
        '2D': { 
            label: 'Instance Normalization 2D',
            description: 'Applies Instance Normalization over a 4D input (N, C, H, W)',
            funcName: 'InstanceNorm2d'
        },
        '3D': { 
            label: 'Instance Normalization 3D',
            description: 'Applies Instance Normalization over a 5D input (N, C, D, H, W)',
            funcName: 'InstanceNorm3d'
        }
    };

    const info = dimensionInfo[dimension];
    
    return {
        label: info.label,
        description: info.description,
        category: 'Normalization',
        moduleType: 'Op',
        params: {
            num_features: commonParams.num_features,
            eps: commonParams.eps,
            momentum: commonParams.momentum,
            affine: { ...commonParams.affine, default: false },
            track_running_stats: { ...commonParams.track_running_stats, default: false }
        },
        toPytorchModule: (params) => {
            return `nn.${info.funcName}(num_features=${params['num_features']}, ` +
                `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
                `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false})`;
        },
        validateInputShape: (inShape, params) => [],
        inferOutputShape: (inShape, params) => [...inShape]
    };
}

// Export individual normalization modules
export const BatchNorm1D = createBatchNormModule('1D');
export const BatchNorm2D = createBatchNormModule('2D');
export const BatchNorm3D = createBatchNormModule('3D');

export const InstanceNorm1D = createInstanceNormModule('1D');
export const InstanceNorm2D = createInstanceNormModule('2D');
export const InstanceNorm3D = createInstanceNormModule('3D');

export const LayerNorm: ModuleDef = {
    label: 'Layer Normalization',
    description: 'Applies Layer Normalization over a mini-batch of inputs',
    category: 'Normalization',
    moduleType: 'Op',
    params: {
        normalized_shape: commonParams.normalized_shape,
        eps: commonParams.eps,
        elementwise_affine: commonParams.elementwise_affine
    },
    toPytorchModule: (params) => {
        return `nn.LayerNorm(normalized_shape=${params['normalized_shape']}, ` +
            `eps=${params['eps'] ?? 1e-5}, elementwise_affine=${params['elementwise_affine'] ?? true})`;
    },
    validateInputShape: (inShape, params) => [],
    inferOutputShape: (inShape, params) => [...inShape]
};

export const GroupNorm: ModuleDef = {
    label: 'Group Normalization',
    description: 'Applies Group Normalization over a mini-batch of inputs',
    category: 'Normalization',
    moduleType: 'Op',
    params: {
        num_groups: commonParams.num_groups,
        num_channels: commonParams.num_channels,
        eps: commonParams.eps,
        affine: commonParams.affine
    },
    toPytorchModule: (params) => {
        return `nn.GroupNorm(num_groups=${params['num_groups']}, num_channels=${params['num_channels']}, ` +
            `eps=${params['eps'] ?? 1e-5}, affine=${params['affine'] ?? true})`;
    },
    validateInputShape: (inShape, params) => [],
    inferOutputShape: (inShape, params) => [...inShape]
};
