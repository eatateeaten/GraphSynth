import { ModuleDef, ParamDef } from './types';

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

// Helper function to calculate pooling output size
function calculatePoolingOutputSize(inputSize: number, kernelSize: number, stride: number, padding: number, dilation: number, ceilMode: boolean): number {
    const numerator = inputSize + 2 * padding - dilation * (kernelSize - 1) - 1;
    const result = numerator / stride + 1;
    return ceilMode ? Math.ceil(result) : Math.floor(result);
}

export const poolingModules: Record<string, ModuleDef> = {
    // Regular pooling layers
    'MaxPool1D': {
        label: 'Max Pool 1D',
        description: 'Applies 1D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.MaxPool1d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, dilation=${params.dilation}, return_indices=${params.return_indices}, ceil_mode=${params.ceil_mode})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 2 && inShape.length !== 3) {
                errors.push(`MaxPool1D requires 2D or 3D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const outputSize = calculatePoolingOutputSize(inShape[inShape.length - 1], params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            return [...inShape.slice(0, -1), outputSize];
        }
    },

    'MaxPool2D': {
        label: 'Max Pool 2D',
        description: 'Applies 2D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.MaxPool2d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, dilation=${params.dilation}, return_indices=${params.return_indices}, ceil_mode=${params.ceil_mode})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 3 && inShape.length !== 4) {
                errors.push(`MaxPool2D requires 3D or 4D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const h = inShape[inShape.length - 2];
            const w = inShape[inShape.length - 1];
            const outH = calculatePoolingOutputSize(h, params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            const outW = calculatePoolingOutputSize(w, params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            return [...inShape.slice(0, -2), outH, outW];
        }
    },

    'MaxPool3D': {
        label: 'Max Pool 3D',
        description: 'Applies 3D max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: maxPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.MaxPool3d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, dilation=${params.dilation}, return_indices=${params.return_indices}, ceil_mode=${params.ceil_mode})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 4 && inShape.length !== 5) {
                errors.push(`MaxPool3D requires 4D or 5D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const d = inShape[inShape.length - 3];
            const h = inShape[inShape.length - 2];
            const w = inShape[inShape.length - 1];
            const outD = calculatePoolingOutputSize(d, params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            const outH = calculatePoolingOutputSize(h, params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            const outW = calculatePoolingOutputSize(w, params.kernel_size, params.stride, params.padding, params.dilation, params.ceil_mode);
            return [...inShape.slice(0, -3), outD, outH, outW];
        }
    },

    'AvgPool1D': {
        label: 'Average Pool 1D',
        description: 'Applies 1D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.AvgPool1d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, ceil_mode=${params.ceil_mode}, count_include_pad=${params.count_include_pad})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 2 && inShape.length !== 3) {
                errors.push(`AvgPool1D requires 2D or 3D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const outputSize = calculatePoolingOutputSize(inShape[inShape.length - 1], params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            return [...inShape.slice(0, -1), outputSize];
        }
    },

    'AvgPool2D': {
        label: 'Average Pool 2D',
        description: 'Applies 2D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPooling2D3DParams,
        emitPytorchModule: (params) => {
            let code = `nn.AvgPool2d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, ceil_mode=${params.ceil_mode}, count_include_pad=${params.count_include_pad}`;
            if (params.divisor_override !== undefined) {
                code += `, divisor_override=${params.divisor_override}`;
            }
            code += ')';
            return code;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 3 && inShape.length !== 4) {
                errors.push(`AvgPool2D requires 3D or 4D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const h = inShape[inShape.length - 2];
            const w = inShape[inShape.length - 1];
            const outH = calculatePoolingOutputSize(h, params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            const outW = calculatePoolingOutputSize(w, params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            return [...inShape.slice(0, -2), outH, outW];
        }
    },

    'AvgPool3D': {
        label: 'Average Pool 3D',
        description: 'Applies 3D average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: avgPooling2D3DParams,
        emitPytorchModule: (params) => {
            let code = `nn.AvgPool3d(kernel_size=${params.kernel_size}, stride=${params.stride}, padding=${params.padding}, ceil_mode=${params.ceil_mode}, count_include_pad=${params.count_include_pad}`;
            if (params.divisor_override !== undefined) {
                code += `, divisor_override=${params.divisor_override}`;
            }
            code += ')';
            return code;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 4 && inShape.length !== 5) {
                errors.push(`AvgPool3D requires 4D or 5D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const d = inShape[inShape.length - 3];
            const h = inShape[inShape.length - 2];
            const w = inShape[inShape.length - 1];
            const outD = calculatePoolingOutputSize(d, params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            const outH = calculatePoolingOutputSize(h, params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            const outW = calculatePoolingOutputSize(w, params.kernel_size, params.stride, params.padding, 1, params.ceil_mode);
            return [...inShape.slice(0, -3), outD, outH, outW];
        }
    },

    'LPPool1D': {
        label: 'LP Pool 1D',
        description: 'Applies 1D power-average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: lpPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.LPPool1d(norm_type=${params.norm_type}, kernel_size=${params.kernel_size}, stride=${params.stride}, ceil_mode=${params.ceil_mode})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 3) {
                errors.push(`LPPool1D requires 3D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const outputSize = calculatePoolingOutputSize(inShape[inShape.length - 1], params.kernel_size, params.stride, 0, 1, params.ceil_mode);
            return [...inShape.slice(0, -1), outputSize];
        }
    },

    'LPPool2D': {
        label: 'LP Pool 2D',
        description: 'Applies 2D power-average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: lpPoolingParams,
        emitPytorchModule: (params) => {
            return `nn.LPPool2d(norm_type=${params.norm_type}, kernel_size=${params.kernel_size}, stride=${params.stride}, ceil_mode=${params.ceil_mode})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 4) {
                errors.push(`LPPool2D requires 4D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            const h = inShape[inShape.length - 2];
            const w = inShape[inShape.length - 1];
            const outH = calculatePoolingOutputSize(h, params.kernel_size, params.stride, 0, 1, params.ceil_mode);
            const outW = calculatePoolingOutputSize(w, params.kernel_size, params.stride, 0, 1, params.ceil_mode);
            return [...inShape.slice(0, -2), outH, outW];
        }
    },

    // Adaptive pooling layers
    'AdaptiveAvgPool1D': {
        label: 'Adaptive Average Pool 1D',
        description: 'Applies 1D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePoolingParams,
        emitPytorchModule: (params) => {
            return `nn.AdaptiveAvgPool1d(output_size=${params.output_size})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 3) {
                errors.push(`AdaptiveAvgPool1D requires 3D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape.slice(0, -1), params.output_size];
        }
    },

    'AdaptiveAvgPool2D': {
        label: 'Adaptive Average Pool 2D',
        description: 'Applies 2D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePooling2D3DParams,
        emitPytorchModule: (params) => {
            const sizeStr = Array.isArray(params.output_size) ? `(${params.output_size.join(', ')})` : params.output_size;
            return `nn.AdaptiveAvgPool2d(output_size=${sizeStr})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 4) {
                errors.push(`AdaptiveAvgPool2D requires 4D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            if (Array.isArray(params.output_size)) {
                return [...inShape.slice(0, -2), ...params.output_size];
            } else {
                return [...inShape.slice(0, -2), params.output_size, params.output_size];
            }
        }
    },

    'AdaptiveAvgPool3D': {
        label: 'Adaptive Average Pool 3D',
        description: 'Applies 3D adaptive average pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: adaptivePooling2D3DParams,
        emitPytorchModule: (params) => {
            const sizeStr = Array.isArray(params.output_size) ? `(${params.output_size.join(', ')})` : params.output_size;
            return `nn.AdaptiveAvgPool3d(output_size=${sizeStr})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 5) {
                errors.push(`AdaptiveAvgPool3D requires 5D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            if (Array.isArray(params.output_size)) {
                return [...inShape.slice(0, -3), ...params.output_size];
            } else {
                return [...inShape.slice(0, -3), params.output_size, params.output_size, params.output_size];
            }
        }
    },

    'AdaptiveMaxPool1D': {
        label: 'Adaptive Max Pool 1D',
        description: 'Applies 1D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePoolingParams, ...adaptiveMaxPoolingParams },
        emitPytorchModule: (params) => {
            return `nn.AdaptiveMaxPool1d(output_size=${params.output_size}, return_indices=${params.return_indices})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 3) {
                errors.push(`AdaptiveMaxPool1D requires 3D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            return [...inShape.slice(0, -1), params.output_size];
        }
    },

    'AdaptiveMaxPool2D': {
        label: 'Adaptive Max Pool 2D',
        description: 'Applies 2D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePooling2D3DParams, ...adaptiveMaxPoolingParams },
        emitPytorchModule: (params) => {
            const sizeStr = Array.isArray(params.output_size) ? `(${params.output_size.join(', ')})` : params.output_size;
            return `nn.AdaptiveMaxPool2d(output_size=${sizeStr}, return_indices=${params.return_indices})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 4) {
                errors.push(`AdaptiveMaxPool2D requires 4D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            if (Array.isArray(params.output_size)) {
                return [...inShape.slice(0, -2), ...params.output_size];
            } else {
                return [...inShape.slice(0, -2), params.output_size, params.output_size];
            }
        }
    },

    'AdaptiveMaxPool3D': {
        label: 'Adaptive Max Pool 3D',
        description: 'Applies 3D adaptive max pooling over an input signal',
        category: 'Pooling',
        moduleType: 'Op',
        params: { ...adaptivePooling2D3DParams, ...adaptiveMaxPoolingParams },
        emitPytorchModule: (params) => {
            const sizeStr = Array.isArray(params.output_size) ? `(${params.output_size.join(', ')})` : params.output_size;
            return `nn.AdaptiveMaxPool3d(output_size=${sizeStr}, return_indices=${params.return_indices})`;
        },
        validateInputShape: (inShape, params) => {
            const errors: string[] = [];
            if (inShape.length !== 5) {
                errors.push(`AdaptiveMaxPool3D requires 5D input tensor, got shape [${inShape.join(', ')}]`);
            }
            return errors;
        },
        inferOutputShape: (inShape, params) => {
            if (Array.isArray(params.output_size)) {
                return [...inShape.slice(0, -3), ...params.output_size];
            } else {
                return [...inShape.slice(0, -3), params.output_size, params.output_size, params.output_size];
            }
        }
    }
};
