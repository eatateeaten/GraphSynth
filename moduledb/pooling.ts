import { ModuleDef, ParamDef } from './types';
import { getTorchCode, forwardShapeInference } from '../OpCompiler/torch_nn_module_op';

// Shared parameter definitions
const commonPoolingParams: Record<string, ParamDef> = {
    kernel_size: {
        label: 'Kernel Size',
        description: 'Size of the pooling window',
        type: 'number',
        required: true
    },
    stride: {
        label: 'Stride',
        description: 'Stride of the pooling operation',
        type: 'number',
        required: false
    },
    padding: {
        label: 'Padding',
        description: 'Implicit zero padding to be added on both sides',
        type: 'number',
        required: false,
        default: 0
    },
    ceil_mode: {
        label: 'Ceil Mode',
        description: 'When True, will use ceil instead of floor to compute the output shape',
        type: 'boolean',
        required: false,
        default: false
    }
};

const maxPoolingParams: Record<string, ParamDef> = {
    ...commonPoolingParams,
    dilation: {
        label: 'Dilation',
        description: 'Spacing between kernel elements',
        type: 'number',
        required: false,
        default: 1
    },
    return_indices: {
        label: 'Return Indices',
        description: 'If True, will return the max indices along with the outputs',
        type: 'boolean',
        required: false,
        default: false
    }
};

const avgPoolingParams: Record<string, ParamDef> = {
    ...commonPoolingParams,
    count_include_pad: {
        label: 'Count Include Pad',
        description: 'When True, will include the zero-padding in the averaging calculation',
        type: 'boolean',
        required: false,
        default: true
    }
};

const avgPooling2D3DParams: Record<string, ParamDef> = {
    ...avgPoolingParams,
    divisor_override: {
        label: 'Divisor Override',
        description: 'If specified, it will be used as divisor, otherwise size of the pooling region will be used',
        type: 'number',
        required: false
    }
};

const lpPoolingParams: Record<string, ParamDef> = {
    norm_type: {
        label: 'Norm Type',
        description: 'Type of norm to use (1, 2, etc.)',
        type: 'number',
        required: true
    },
    kernel_size: commonPoolingParams.kernel_size,
    stride: commonPoolingParams.stride,
    ceil_mode: commonPoolingParams.ceil_mode
};

const adaptivePoolingParams: Record<string, ParamDef> = {
    output_size: {
        label: 'Output Size',
        description: 'Target output size',
        type: 'number',
        required: true
    }
};

const adaptivePooling2D3DParams: Record<string, ParamDef> = {
    output_size: {
        label: 'Output Size',
        description: 'Target output size',
        type: 'shape',
        required: true
    }
};

const adaptiveMaxPoolingParams: Record<string, ParamDef> = {
    return_indices: {
        label: 'Return Indices',
        description: 'If True, will return the max indices along with the outputs',
        type: 'boolean',
        required: false,
        default: false
    }
};

export const poolingModules: Record<string, ModuleDef> = {
    // Regular pooling layers
    'MaxPool1D': {
        label: 'Max Pool 1D',
        description: 'Applies 1D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        toPytorchModule: (params) => getTorchCode('MaxPool1D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 2 && inShape.length !== 3) {
                throw new Error(`MaxPool1D requires 2D or 3D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('MaxPool1D', inShape, params)
    },

    'MaxPool2D': {
        label: 'Max Pool 2D',
        description: 'Applies 2D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        toPytorchModule: (params) => getTorchCode('MaxPool2D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 3 && inShape.length !== 4) {
                throw new Error(`MaxPool2D requires 3D or 4D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('MaxPool2D', inShape, params)
    },

    'MaxPool3D': {
        label: 'Max Pool 3D',
        description: 'Applies 3D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        toPytorchModule: (params) => getTorchCode('MaxPool3D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4 && inShape.length !== 5) {
                throw new Error(`MaxPool3D requires 4D or 5D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('MaxPool3D', inShape, params)
    },

    'AvgPool1D': {
        label: 'Average Pool 1D',
        description: 'Applies 1D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPoolingParams,
        toPytorchModule: (params) => getTorchCode('AvgPool1D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 2 && inShape.length !== 3) {
                throw new Error(`AvgPool1D requires 2D or 3D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AvgPool1D', inShape, params)
    },

    'AvgPool2D': {
        label: 'Average Pool 2D',
        description: 'Applies 2D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPooling2D3DParams,
        toPytorchModule: (params) => getTorchCode('AvgPool2D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 3 && inShape.length !== 4) {
                throw new Error(`AvgPool2D requires 3D or 4D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AvgPool2D', inShape, params)
    },

    'AvgPool3D': {
        label: 'Average Pool 3D',
        description: 'Applies 3D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPooling2D3DParams,
        toPytorchModule: (params) => getTorchCode('AvgPool3D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4 && inShape.length !== 5) {
                throw new Error(`AvgPool3D requires 4D or 5D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AvgPool3D', inShape, params)
    },

    'LPPool1D': {
        label: 'LP Pool 1D',
        description: 'Applies 1D power-average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: lpPoolingParams,
        toPytorchModule: (params) => getTorchCode('LPPool1D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 3) {
                throw new Error(`LPPool1D requires 3D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('LPPool1D', inShape, params)
    },

    'LPPool2D': {
        label: 'LP Pool 2D',
        description: 'Applies 2D power-average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: lpPoolingParams,
        toPytorchModule: (params) => getTorchCode('LPPool2D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4) {
                throw new Error(`LPPool2D requires 4D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('LPPool2D', inShape, params)
    },

    // Adaptive pooling layers
    'AdaptiveAvgPool1D': {
        label: 'Adaptive Average Pool 1D',
        description: 'Applies 1D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePoolingParams,
        toPytorchModule: (params) => getTorchCode('AdaptiveAvgPool1D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 3) {
                throw new Error(`AdaptiveAvgPool1D requires 3D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveAvgPool1D', inShape, params)
    },

    'AdaptiveAvgPool2D': {
        label: 'Adaptive Average Pool 2D',
        description: 'Applies 2D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePooling2D3DParams,
        toPytorchModule: (params) => getTorchCode('AdaptiveAvgPool2D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4) {
                throw new Error(`AdaptiveAvgPool2D requires 4D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveAvgPool2D', inShape, params)
    },

    'AdaptiveAvgPool3D': {
        label: 'Adaptive Average Pool 3D',
        description: 'Applies 3D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePooling2D3DParams,
        toPytorchModule: (params) => getTorchCode('AdaptiveAvgPool3D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 5) {
                throw new Error(`AdaptiveAvgPool3D requires 5D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveAvgPool3D', inShape, params)
    },

    'AdaptiveMaxPool1D': {
        label: 'Adaptive Max Pool 1D',
        description: 'Applies 1D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePoolingParams, ...adaptiveMaxPoolingParams },
        toPytorchModule: (params) => getTorchCode('AdaptiveMaxPool1D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 3) {
                throw new Error(`AdaptiveMaxPool1D requires 3D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveMaxPool1D', inShape, params)
    },

    'AdaptiveMaxPool2D': {
        label: 'Adaptive Max Pool 2D',
        description: 'Applies 2D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePooling2D3DParams, ...adaptiveMaxPoolingParams },
        toPytorchModule: (params) => getTorchCode('AdaptiveMaxPool2D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 4) {
                throw new Error(`AdaptiveMaxPool2D requires 4D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveMaxPool2D', inShape, params)
    },

    'AdaptiveMaxPool3D': {
        label: 'Adaptive Max Pool 3D',
        description: 'Applies 3D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePooling2D3DParams, ...adaptiveMaxPoolingParams },
        toPytorchModule: (params) => getTorchCode('AdaptiveMaxPool3D', params),
        validateInputShape: (inShape, params) => {
            if (inShape.length !== 5) {
                throw new Error(`AdaptiveMaxPool3D requires 5D input tensor, got shape ${inShape}`);
            }
            return [];
        },
        inferOutputShape: (inShape, params) => forwardShapeInference('AdaptiveMaxPool3D', inShape, params)
    }
};
