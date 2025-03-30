interface ModuleMetadata {
    required_params: string[];
    optional_params: string[];
    code_generator: (params: Record<string, any>) => string;
    forward_shape_inference: (inShape: number[], params: Record<string, any>) => number[];
}

// Helper functions for shape inference
function convOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number
): number {
    // out = floor((inSize + 2*padding - dilation*(kernelSize - 1) - 1)/ stride + 1)
    return Math.floor((inSize + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1);
}

function convTransposeOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number,
    outputPadding: number
): number {
    // out = (inSize - 1)*stride - 2*padding + dilation*(kernelSize - 1) + outputPadding + 1
    return (
        (inSize - 1) * stride -
        2 * padding +
        dilation * (kernelSize - 1) +
        outputPadding +
        1
    );
}

function poolOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number,
    ceil_mode: boolean
): number {
    // out = floor((inSize + 2*padding - dilation*(kernelSize - 1) - 1)/ stride + 1)
    // If ceil_mode, PyTorch uses math.ceil instead of math.floor.
    const raw = (inSize + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1;
    return ceil_mode ? Math.ceil(raw) : Math.floor(raw);
}

function ensureArray(val: number | number[]): number[] {
    // Utility to ensure we always treat output_size as an array.
    if (Array.isArray(val)) {
        return val;
    } else {
        return [val];
    }
}

export const nn_module_metadata: Record<string, ModuleMetadata> = {
    // Linear layers
    "Linear": {
        required_params: ["input_features", "output_features"],
        optional_params: ["bias"],
        code_generator: (params) => `nn.Linear(${params['input_features']}, ${params['output_features']}, bias=${params['bias'] ?? true})`,
        forward_shape_inference: (inShape, params) => {
            // Typically, linear expects [N, input_features], output => [N, output_features]
            // We'll assume inShape is [N, input_features]
            return [inShape[0], params['output_features']];
        },
    },

    // Convolutional layers
    "Conv1D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.Conv1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // inShape = [N, C_in, L]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const L_out = convOutputSize(inShape[2], kernel, stride, padding, dilation);
            return [inShape[0], params['out_channels'], L_out];
        },
    },

    "Conv2D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.Conv2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // inShape = [N, C_in, H_in, W_in]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const H_out = convOutputSize(inShape[2], kernel, stride, padding, dilation);
            const W_out = convOutputSize(inShape[3], kernel, stride, padding, dilation);
            return [inShape[0], params['out_channels'], H_out, W_out];
        },
    },

    "Conv3D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.Conv3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // inShape = [N, C_in, D_in, H_in, W_in]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const D_out = convOutputSize(inShape[2], kernel, stride, padding, dilation);
            const H_out = convOutputSize(inShape[3], kernel, stride, padding, dilation);
            const W_out = convOutputSize(inShape[4], kernel, stride, padding, dilation);
            return [inShape[0], params['out_channels'], D_out, H_out, W_out];
        },
    },

    // Transposed convolutions
    "ConvTranspose1D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // [N, C_in, L_in] -> [N, out_channels, L_out]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const L_out = convTransposeOutputSize(inShape[2], kernel, stride, padding, dilation, outputPadding);
            return [inShape[0], params['out_channels'], L_out];
        },
    },

    "ConvTranspose2D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // [N, C_in, H_in, W_in] -> [N, out_channels, H_out, W_out]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const H_out = convTransposeOutputSize(inShape[2], kernel, stride, padding, dilation, outputPadding);
            const W_out = convTransposeOutputSize(inShape[3], kernel, stride, padding, dilation, outputPadding);
            return [inShape[0], params['out_channels'], H_out, W_out];
        },
    },

    "ConvTranspose3D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // [N, C_in, D_in, H_in, W_in] -> [N, out_channels, D_out, H_out, W_out]
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            const D_out = convTransposeOutputSize(inShape[2], kernel, stride, padding, dilation, outputPadding);
            const H_out = convTransposeOutputSize(inShape[3], kernel, stride, padding, dilation, outputPadding);
            const W_out = convTransposeOutputSize(inShape[4], kernel, stride, padding, dilation, outputPadding);
            return [inShape[0], params['out_channels'], D_out, H_out, W_out];
        },
    },

    // Pooling layers
    "MaxPool1D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool1d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, L] -> [N, C, L_out]
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const L_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], L_out];
        },
    },

    "MaxPool2D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool2d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, H, W] -> [N, C, H_out, W_out]
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const H_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], H_out, W_out];
        },
    },

    "MaxPool3D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool3d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, D, H, W] -> [N, C, D_out, H_out, W_out]
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;

            const D_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const H_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[4], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], D_out, H_out, W_out];
        },
    },

    "AvgPool1D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad"],
        code_generator: (params) => `nn.AvgPool1d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true})`,
        forward_shape_inference: (inShape, params) => {
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            // dilation is not an arg for AvgPool in PyTorch, default = 1
            const dilation = 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const L_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], L_out];
        },
    },

    "AvgPool2D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        code_generator: (params) => `nn.AvgPool2d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true}, ` +
            `divisor_override=${params['divisor_override'] ?? 'None'})`,
        forward_shape_inference: (inShape, params) => {
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const H_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], H_out, W_out];
        },
    },

    "AvgPool3D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        code_generator: (params) => `nn.AvgPool3d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true}, ` +
            `divisor_override=${params['divisor_override'] ?? 'None'})`,
        forward_shape_inference: (inShape, params) => {
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const D_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const H_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[4], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], D_out, H_out, W_out];
        },
    },

    "LPPool1D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool1d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, L] -> [N, C, L_out]
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = 0; // not in signature
            const dilation = 1; // not in signature
            const ceil_mode = params['ceil_mode'] ?? false;
            const L_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], L_out];
        },
    },

    "LPPool2D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool2d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, H, W] -> [N, C, H_out, W_out]
            const kernel = params['kernel_size'];
            const stride = (params['stride'] !== undefined) ? params['stride'] : kernel;
            const padding = 0;
            const dilation = 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            const H_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            return [inShape[0], inShape[1], H_out, W_out];
        },
    },

    // Adaptive pooling
    "AdaptiveAvgPool1D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool1d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, L_in] -> [N, C, output_size]
            return [inShape[0], inShape[1], params['output_size']];
        },
    },

    "AdaptiveAvgPool2D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool2d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, H_in, W_in] -> [N, C, outH, outW]
            const outSize = ensureArray(params['output_size']);
            if (outSize.length === 1) {
                // e.g. output_size=H
                return [inShape[0], inShape[1], outSize[0], outSize[0]];
            } else {
                // e.g. output_size=[H, W]
                return [inShape[0], inShape[1], outSize[0], outSize[1]];
            }
        },
    },

    "AdaptiveAvgPool3D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool3d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, D_in, H_in, W_in] -> [N, C, outD, outH, outW]
            const outSize = ensureArray(params['output_size']);
            // Could be single, double, or triple, but typically triple
            if (outSize.length === 1) {
                return [inShape[0], inShape[1], outSize[0], outSize[0], outSize[0]];
            } else if (outSize.length === 2) {
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[1]];
            } else {
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[2]];
            }
        },
    },

    "AdaptiveMaxPool1D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool1d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, L_in] -> [N, C, output_size]
            return [inShape[0], inShape[1], params['output_size']];
        },
    },

    "AdaptiveMaxPool2D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool2d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, H_in, W_in] -> [N, C, outH, outW]
            const outSize = ensureArray(params['output_size']);
            if (outSize.length === 1) {
                return [inShape[0], inShape[1], outSize[0], outSize[0]];
            } else {
                return [inShape[0], inShape[1], outSize[0], outSize[1]];
            }
        },
    },

    "AdaptiveMaxPool3D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool3d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // [N, C, D_in, H_in, W_in] -> [N, C, outD, outH, outW]
            const outSize = ensureArray(params['output_size']);
            if (outSize.length === 1) {
                return [inShape[0], inShape[1], outSize[0], outSize[0], outSize[0]];
            } else if (outSize.length === 2) {
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[1]];
            } else {
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[2]];
            }
        },
    },

    // Normalization layers
    "BatchNorm1D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.BatchNorm1d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true}${dimParam})`;
        },
        forward_shape_inference: (inShape) => {
            // [N, C, L] => same shape
            return inShape;
        },
    },

    "BatchNorm2D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.BatchNorm2d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true}${dimParam})`;
        },
        forward_shape_inference: (inShape) => {
            // [N, C, H, W] => same shape
            return inShape;
        },
    },

    "BatchNorm3D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.BatchNorm3d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true}${dimParam})`;
        },
        forward_shape_inference: (inShape) => {
            // [N, C, D, H, W] => same shape
            return inShape;
        },
    },

    "LayerNorm": {
        required_params: ["normalized_shape"],
        optional_params: ["eps", "elementwise_affine", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.LayerNorm(normalized_shape=${params['normalized_shape']}, ` +
            `eps=${params['eps'] ?? 1e-5}, elementwise_affine=${params['elementwise_affine'] ?? true}${dimParam})`;
        },
        forward_shape_inference: (inShape) => {
            // shape unchanged
            return inShape;
        },
    },

    "GroupNorm": {
        required_params: ["num_groups", "num_channels"],
        optional_params: ["eps", "affine", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.GroupNorm(num_groups=${params['num_groups']}, num_channels=${params['num_channels']}, ` +
            `eps=${params['eps'] ?? 1e-5}, affine=${params['affine'] ?? true}${dimParam})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "InstanceNorm1D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.InstanceNorm1d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false}${dimParam})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "InstanceNorm2D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.InstanceNorm2d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false}${dimParam})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "InstanceNorm3D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats", "dim"],
        code_generator: (params) => {
            const dimParam = params['dim'] !== undefined ? `, dim=${params['dim']}` : '';
            return `nn.InstanceNorm3d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false}${dimParam})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    // Dropout layers
    "Dropout": {
        required_params: [],
        optional_params: ["p", "inplace", "dim"],
        code_generator: (params) => {
            // Standard Dropout applies to all dimensions randomly
            // If dim is provided, we use functional form to target specific dimension
            if (params['dim'] !== undefined) {
                return `nn.functional.dropout(input, p=${params['p'] ?? 0.5}, training=self.training, inplace=${params['inplace'] ?? false}, dim=${params['dim']})`;
            }
            return `nn.Dropout(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "Dropout2D": {
        required_params: [],
        optional_params: ["p", "inplace", "dim"],
        code_generator: (params) => {
            // Dropout2d normally zeros entire channels (dim=1 for NCHW)
            // But we can allow user to specify which dimension to drop 
            if (params['dim'] !== undefined) {
                return `nn.functional.dropout2d(input, p=${params['p'] ?? 0.5}, training=self.training, inplace=${params['inplace'] ?? false}, dim=${params['dim']})`;
            }
            return `nn.Dropout2d(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "Dropout3D": {
        required_params: [],
        optional_params: ["p", "inplace", "dim"],
        code_generator: (params) => {
            // Dropout3d normally zeros entire volumetric features (dim=1 for NCDHW)
            // But we can allow user to specify which dimension to drop
            if (params['dim'] !== undefined) {
                return `nn.functional.dropout3d(input, p=${params['p'] ?? 0.5}, training=self.training, inplace=${params['inplace'] ?? false}, dim=${params['dim']})`;
            }
            return `nn.Dropout3d(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    "AlphaDropout": {
        required_params: [],
        optional_params: ["p", "inplace", "dim"],
        code_generator: (params) => {
            // AlphaDropout for SELU activations
            if (params['dim'] !== undefined) {
                // Note: AlphaDropout doesn't have a direct dimension-targeted version in PyTorch
                // This is a custom implementation suggestion
                return `nn.functional.alpha_dropout(input, p=${params['p'] ?? 0.5}, training=self.training, inplace=${params['inplace'] ?? false}, dim=${params['dim']})`;
            }
            return `nn.AlphaDropout(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`;
        },
        forward_shape_inference: (inShape) => inShape,
    },

    // Activation functions
    "ReLU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.ReLU(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "LeakyReLU": {
        required_params: [],
        optional_params: ["negative_slope", "inplace"],
        code_generator: (params) => `nn.LeakyReLU(negative_slope=${params['negative_slope'] ?? 0.01}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Sigmoid": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Sigmoid()`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Tanh": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Tanh()`,
        forward_shape_inference: (inShape) => inShape,
    },

    "ELU": {
        required_params: [],
        optional_params: ["alpha", "inplace"],
        code_generator: (params) => `nn.ELU(alpha=${params['alpha'] ?? 1.0}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "SELU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.SELU(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "CELU": {
        required_params: [],
        optional_params: ["alpha", "inplace"],
        code_generator: (params) => `nn.CELU(alpha=${params['alpha'] ?? 1.0}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "GELU": {
        required_params: [],
        optional_params: ["approximate"],
        code_generator: (params) => `nn.GELU(approximate='${params['approximate'] ?? 'none'}')`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Softplus": {
        required_params: [],
        optional_params: ["beta", "threshold"],
        code_generator: (params) => `nn.Softplus(beta=${params['beta'] ?? 1}, threshold=${params['threshold'] ?? 20})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Softsign": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Softsign()`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Softmax": {
        required_params: [],
        optional_params: ["dim"],
        code_generator: (params) => `nn.Softmax(dim=${params['dim'] ?? -1})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "LogSoftmax": {
        required_params: [],
        optional_params: ["dim"],
        code_generator: (params) => `nn.LogSoftmax(dim=${params['dim'] ?? -1})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "PReLU": {
        required_params: [],
        optional_params: ["num_parameters", "init"],
        code_generator: (params) => `nn.PReLU(num_parameters=${params['num_parameters'] ?? 1}, init=${params['init'] ?? 0.25})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Hardtanh": {
        required_params: [],
        optional_params: ["min_val", "max_val", "inplace"],
        code_generator: (params) => `nn.Hardtanh(min_val=${params['min_val'] ?? -1.0}, max_val=${params['max_val'] ?? 1.0}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Hardshrink": {
        required_params: [],
        optional_params: ["lambd"],
        code_generator: (params) => `nn.Hardshrink(lambd=${params['lambd'] ?? 0.5})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Hardsigmoid": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Hardsigmoid(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Hardswish": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Hardswish(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "RReLU": {
        required_params: [],
        optional_params: ["lower", "upper", "inplace"],
        code_generator: (params) => `nn.RReLU(lower=${params['lower'] ?? 1/8}, upper=${params['upper'] ?? 1/3}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Softshrink": {
        required_params: [],
        optional_params: ["lambd"],
        code_generator: (params) => `nn.Softshrink(lambd=${params['lambd'] ?? 0.5})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Tanhshrink": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Tanhshrink()`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Threshold": {
        required_params: ["threshold", "value"],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Threshold(threshold=${params['threshold']}, value=${params['value']}, inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "ReLU6": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.ReLU6(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "SiLU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.SiLU(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    "Mish": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Mish(inplace=${params['inplace'] ?? false})`,
        forward_shape_inference: (inShape) => inShape,
    },

    // Reshape operations
    "Reshape": {
        required_params: ["shape"],
        optional_params: [],
        code_generator: (params) => `torch.reshape(${params['shape'].join(', ')})`,
        forward_shape_inference: (inShape, params) => {
            // We'll assume the user-provided shape is correct
            // If shape includes -1, normally we'd compute it, but let's assume user does that.
            return params['shape'];
        },
    },

    "Permute": {
        required_params: ["dims"],
        optional_params: [],
        code_generator: (params) => `torch.permute(${params['dims'].join(', ')})`,
        forward_shape_inference: (inShape, params) => {
            const dims = params['dims'];
            const outShape = dims.map((d: number) => inShape[d]);
            return outShape;
        },
    },

    "Flatten": {
        required_params: [],
        optional_params: ["start_dim", "end_dim"],
        code_generator: (params) => `nn.Flatten(start_dim=${params['start_dim'] ?? 1}, end_dim=${params['end_dim'] ?? -1})`,
        forward_shape_inference: (inShape, params) => {
            const start = params['start_dim'] ?? 1;
            let end = params['end_dim'] ?? -1;
            if (end < 0) {
                end = inShape.length + end;
            }
            const before = inShape.slice(0, start);
            const middle = inShape.slice(start, end + 1).reduce((acc, val) => acc * val, 1);
            const after = inShape.slice(end + 1);
            return [...before, middle, ...after];
        },
    },

    "Unflatten": {
        required_params: ["dim", "unflattened_size"],
        optional_params: [],
        code_generator: (params) => `nn.Unflatten(dim=${params['dim']}, unflattened_size=${params['unflattened_size']})`,
        forward_shape_inference: (inShape, params) => {
            // we replace inShape[dim] with the dimensions in unflattened_size
            // e.g. if inShape = [N, 100], dim=1, unflattened_size=[10,10] => [N,10,10]
            const outShape = [...inShape];
            const dim = params['dim'];
            const unflat = params['unflattened_size'];
            outShape.splice(dim, 1, ...unflat);
            return outShape;
        },
    },

    // Identity operation (simply passes input through unchanged)
    "Identity": {
        required_params: [],
        optional_params: [],
        code_generator: (_) => `nn.Identity()`,
        forward_shape_inference: (inShape) => inShape,
    },
};

export function validateParams(module_type: string, params: Record<string, any>): boolean {
    if (!(module_type in nn_module_metadata)) {
        throw new Error(`Unknown module type: ${module_type}`);
    }

    const metadata = nn_module_metadata[module_type];
    const missing_params: string[] = [];

    for (const param of metadata.required_params) {
        if (!(param in params)) {
            missing_params.push(param);
        }
    }

    if (missing_params.length > 0) {
        throw new Error(`Missing required parameters for ${module_type}: ${missing_params.join(', ')}`);
    }

    return true;
}

export function getTorchCode(module_type: string, params: Record<string, any>): string {
    validateParams(module_type, params);

    return nn_module_metadata[module_type].code_generator(params);
}

export function forwardShapeInference(module_type: string, inShape: number[], params: Record<string, any>): number[] {
    validateParams(module_type, params);

    let outShape: number[];
    try {
        outShape = nn_module_metadata[module_type].forward_shape_inference(inShape, params);
    } catch (e: any) {
        throw new Error(`Shape inference error for ${module_type} with inShape=${JSON.stringify(inShape)}: ${e.message}`);
    }

    // Additional check: if any dimension is <= 0, consider it invalid
    if (outShape.some(dim => dim <= 0)) {
        throw new Error(`Inferred an invalid shape: [${outShape.join(', ')}]. Dimensions must be > 0.`);
    }

    return outShape;
}

// Add getElementwiseOpCode to ensure backward compatibility
export function getElementwiseOpCode(opType: string, input?: string): string {
    switch (opType.toLowerCase()) {
        case 'add':
            return 'torch.add';
        case 'sub':
            return 'torch.sub';
        case 'mul':
            return 'torch.mul';
        case 'div':
            return 'torch.div';
        case 'pow':
            return 'torch.pow';
        case 'min':
            return 'torch.min';
        case 'max':
            return 'torch.max';
        case 'and':
            return 'torch.logical_and';
        case 'or':
            return 'torch.logical_or';
        case 'xor':
            return 'torch.logical_xor';
        case 'not':
            return 'torch.logical_not';
        case 'eq':
            return 'torch.eq';
        case 'ne':
            return 'torch.ne';
        case 'lt':
            return 'torch.lt';
        case 'le':
            return 'torch.le';
        case 'gt':
            return 'torch.gt';
        case 'ge':
            return 'torch.ge';
        case 'identity':
            return input ? input : 'x => x'; // Return input unchanged
        default:
            throw new Error(`Unknown elementwise operation type: ${opType}`);
    }
}
