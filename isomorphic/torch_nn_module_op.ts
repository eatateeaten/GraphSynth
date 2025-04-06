import { assert } from "./utils";

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
            /* From torch documentation:
             * Input: (∗,Hin) where ∗ means any number of dimensions including none and Hin = in_features
             * Output: (∗,Hout) where all but the last dimension are the same shape as the input and Hout = out_features */
            assert(inShape.length >= 1, "Linear requires at least 1D input");
            assert(inShape[inShape.length-1] === params.input_features, "Last dimension must be equal to input features");
            return [...inShape.slice(0, -1), params.output_features];
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
            // Conv1D requires 2D [C_in, L] or 3D [N, C_in, L] input
            const is3D = inShape.length === 3;
            const is2D = inShape.length === 2;
            
            if (!is3D && !is2D) {
                throw new Error(`Conv1D requires 2D or 3D input tensor, got shape ${inShape}`);
            }
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is3D ? 1 : 0;
            if (inShape[channelDim] !== params['in_channels']) {
                throw new Error(`Conv1D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            }
            
            // Calculate output spatial dimension
            const lenDim = is3D ? 2 : 1;
            const L_out = convOutputSize(inShape[lenDim], kernel, stride, padding, dilation);
            
            if (L_out <= 0) {
                throw new Error(`Conv1D output length would be ${L_out}, which is invalid`);
            }
            
            // Return appropriate shape based on input rank
            if (is3D) {
                return [inShape[0], params['out_channels'], L_out];
            } else {
                return [params['out_channels'], L_out];
            }
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
            // Conv2D requires 3D [C_in, H, W] or 4D [N, C_in, H, W] input
            const is4D = inShape.length === 4;
            const is3D = inShape.length === 3;
            
            if (!is4D && !is3D) {
                throw new Error(`Conv2D requires 3D or 4D input tensor, got shape ${inShape}`);
            }
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is4D ? 1 : 0;
            if (inShape[channelDim] !== params['in_channels']) {
                throw new Error(`Conv2D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            }
            
            // Calculate output spatial dimensions
            const hDim = is4D ? 2 : 1;
            const wDim = is4D ? 3 : 2;
            
            const H_out = convOutputSize(inShape[hDim], kernel, stride, padding, dilation);
            const W_out = convOutputSize(inShape[wDim], kernel, stride, padding, dilation);
            
            if (H_out <= 0 || W_out <= 0) {
                throw new Error(`Conv2D output dimensions would be ${H_out}x${W_out}, which is invalid`);
            }
            
            // Return appropriate shape based on input rank
            if (is4D) {
                return [inShape[0], params['out_channels'], H_out, W_out];
            } else {
                return [params['out_channels'], H_out, W_out];
            }
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
            // Conv3D requires 4D [C_in, D, H, W] or 5D [N, C_in, D, H, W] input
            const is5D = inShape.length === 5;
            const is4D = inShape.length === 4;
            
            assert(is5D || is4D, `Conv3D requires 4D or 5D input tensor, got shape ${inShape}`);
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is5D ? 1 : 0;
            assert(inShape[channelDim] === params['in_channels'], 
                `Conv3D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            
            // Calculate output spatial dimensions
            const dDim = is5D ? 2 : 1;
            const hDim = is5D ? 3 : 2;
            const wDim = is5D ? 4 : 3;
            
            const D_out = convOutputSize(inShape[dDim], kernel, stride, padding, dilation);
            const H_out = convOutputSize(inShape[hDim], kernel, stride, padding, dilation);
            const W_out = convOutputSize(inShape[wDim], kernel, stride, padding, dilation);
            
            assert(D_out > 0 && H_out > 0 && W_out > 0, 
                `Conv3D output dimensions would be ${D_out}x${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is5D) {
                return [inShape[0], params['out_channels'], D_out, H_out, W_out];
            } else {
                return [params['out_channels'], D_out, H_out, W_out];
            }
        },
    },

    // Transposed convolutions
    "ConvTranspose1D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
            `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // ConvTranspose1D requires 2D [C_in, L] or 3D [N, C_in, L] input
            const is3D = inShape.length === 3;
            const is2D = inShape.length === 2;
            
            assert(is3D || is2D, `ConvTranspose1D requires 2D or 3D input tensor, got shape ${inShape}`);
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is3D ? 1 : 0;
            assert(inShape[channelDim] === params['in_channels'], 
                `ConvTranspose1D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            
            // Calculate output spatial dimension
            const lenDim = is3D ? 2 : 1;
            const L_out = convTransposeOutputSize(inShape[lenDim], kernel, stride, padding, dilation, outputPadding);
            
            assert(L_out > 0, `ConvTranspose1D output length would be ${L_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is3D) {
                return [inShape[0], params['out_channels'], L_out];
            } else {
                return [params['out_channels'], L_out];
            }
        },
    },

    "ConvTranspose2D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
            `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // ConvTranspose2D requires 3D [C_in, H, W] or 4D [N, C_in, H, W] input
            const is4D = inShape.length === 4;
            const is3D = inShape.length === 3;
            
            assert(is4D || is3D, `ConvTranspose2D requires 3D or 4D input tensor, got shape ${inShape}`);
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is4D ? 1 : 0;
            assert(inShape[channelDim] === params['in_channels'], 
                `ConvTranspose2D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            
            // Calculate output spatial dimensions
            const hDim = is4D ? 2 : 1;
            const wDim = is4D ? 3 : 2;
            
            const H_out = convTransposeOutputSize(inShape[hDim], kernel, stride, padding, dilation, outputPadding);
            const W_out = convTransposeOutputSize(inShape[wDim], kernel, stride, padding, dilation, outputPadding);
            
            assert(H_out > 0 && W_out > 0, 
                `ConvTranspose2D output dimensions would be ${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is4D) {
                return [inShape[0], params['out_channels'], H_out, W_out];
            } else {
                return [params['out_channels'], H_out, W_out];
            }
        },
    },

    "ConvTranspose3D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) =>
            `nn.ConvTranspose3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
            `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
        forward_shape_inference: (inShape, params) => {
            // ConvTranspose3D requires 4D [C_in, D, H, W] or 5D [N, C_in, D, H, W] input
            const is5D = inShape.length === 5;
            const is4D = inShape.length === 4;
            
            assert(is5D || is4D, `ConvTranspose3D requires 4D or 5D input tensor, got shape ${inShape}`);
            
            // Extract parameters with defaults
            const stride = params['stride'] ?? 1;
            const padding = params['padding'] ?? 0;
            const outputPadding = params['output_padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const kernel = params['kernel_size'];
            
            // Validate channels
            const channelDim = is5D ? 1 : 0;
            assert(inShape[channelDim] === params['in_channels'], 
                `ConvTranspose3D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
            
            // Calculate output spatial dimensions
            const dDim = is5D ? 2 : 1;
            const hDim = is5D ? 3 : 2;
            const wDim = is5D ? 4 : 3;
            
            const D_out = convTransposeOutputSize(inShape[dDim], kernel, stride, padding, dilation, outputPadding);
            const H_out = convTransposeOutputSize(inShape[hDim], kernel, stride, padding, dilation, outputPadding);
            const W_out = convTransposeOutputSize(inShape[wDim], kernel, stride, padding, dilation, outputPadding);
            
            assert(D_out > 0 && H_out > 0 && W_out > 0, 
                `ConvTranspose3D output dimensions would be ${D_out}x${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is5D) {
                return [inShape[0], params['out_channels'], D_out, H_out, W_out];
            } else {
                return [params['out_channels'], D_out, H_out, W_out];
            }
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
            // MaxPool1D requires 2D [C, L] or 3D [N, C, L] input
            const is3D = inShape.length === 3;
            const is2D = inShape.length === 2;
            
            if (!is3D && !is2D) {
                throw new Error(`MaxPool1D requires 2D or 3D input tensor, got shape ${inShape}`);
            }
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output length
            const lenDim = is3D ? 2 : 1;
            const L_out = poolOutputSize(inShape[lenDim], kernel, stride, padding, dilation, ceil_mode);
            
            if (L_out <= 0) {
                throw new Error(`MaxPool1D output length would be ${L_out}, which is invalid`);
            }
            
            // Return output shape based on input rank
            if (is3D) {
                return [inShape[0], inShape[1], L_out];
            } else {
                return [inShape[0], L_out];
            }
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
            // MaxPool2D requires 3D [C, H, W] or 4D [N, C, H, W] input
            const is4D = inShape.length === 4;
            const is3D = inShape.length === 3;
            
            if (!is4D && !is3D) {
                throw new Error(`MaxPool2D requires 3D or 4D input tensor, got shape ${inShape}`);
            }
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output spatial dimensions
            const hDim = is4D ? 2 : 1;
            const wDim = is4D ? 3 : 2;
            
            const H_out = poolOutputSize(inShape[hDim], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[wDim], kernel, stride, padding, dilation, ceil_mode);
            
            if (H_out <= 0 || W_out <= 0) {
                throw new Error(`MaxPool2D output dimensions would be ${H_out}x${W_out}, which is invalid`);
            }
            
            // Return appropriate shape based on input rank
            const outShape = [...inShape];
            outShape[hDim] = H_out;
            outShape[wDim] = W_out;
            
            return outShape;
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
            // MaxPool3D requires 4D [C, D, H, W] or 5D [N, C, D, H, W] input
            const is5D = inShape.length === 5;
            const is4D = inShape.length === 4;
            
            assert(is5D || is4D, `MaxPool3D requires 4D or 5D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = params['dilation'] ?? 1;
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output spatial dimensions
            const dDim = is5D ? 2 : 1;
            const hDim = is5D ? 3 : 2;
            const wDim = is5D ? 4 : 3;
            
            const D_out = poolOutputSize(inShape[dDim], kernel, stride, padding, dilation, ceil_mode);
            const H_out = poolOutputSize(inShape[hDim], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[wDim], kernel, stride, padding, dilation, ceil_mode);
            
            assert(D_out > 0 && H_out > 0 && W_out > 0, 
                `MaxPool3D output dimensions would be ${D_out}x${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            const outShape = [...inShape];
            outShape[dDim] = D_out;
            outShape[hDim] = H_out;
            outShape[wDim] = W_out;
            
            return outShape;
        },
    },

    "AvgPool1D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad"],
        code_generator: (params) => `nn.AvgPool1d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true})`,
        forward_shape_inference: (inShape, params) => {
            // AvgPool1D requires 2D [C, L] or 3D [N, C, L] input
            const is3D = inShape.length === 3;
            const is2D = inShape.length === 2;
            
            assert(is3D || is2D, `AvgPool1D requires 2D or 3D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = 1; // dilation is not an arg for AvgPool in PyTorch
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output length
            const lenDim = is3D ? 2 : 1;
            const L_out = poolOutputSize(inShape[lenDim], kernel, stride, padding, dilation, ceil_mode);
            
            assert(L_out > 0, `AvgPool1D output length would be ${L_out}, which is invalid`);
            
            // Return output shape based on input rank
            if (is3D) {
                return [inShape[0], inShape[1], L_out];
            } else {
                return [inShape[0], L_out];
            }
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
            // AvgPool2D requires 3D [C, H, W] or 4D [N, C, H, W] input
            const is4D = inShape.length === 4;
            const is3D = inShape.length === 3;
            
            assert(is4D || is3D, `AvgPool2D requires 3D or 4D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = 1; // dilation is not an arg for AvgPool in PyTorch
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output spatial dimensions
            const hDim = is4D ? 2 : 1;
            const wDim = is4D ? 3 : 2;
            
            const H_out = poolOutputSize(inShape[hDim], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[wDim], kernel, stride, padding, dilation, ceil_mode);
            
            assert(H_out > 0 && W_out > 0, `AvgPool2D output dimensions would be ${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is4D) {
                return [inShape[0], inShape[1], H_out, W_out];
            } else {
                return [inShape[0], H_out, W_out];
            }
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
            // AvgPool3D requires 4D [C, D, H, W] or 5D [N, C, D, H, W] input
            const is5D = inShape.length === 5;
            const is4D = inShape.length === 4;
            
            assert(is5D || is4D, `AvgPool3D requires 4D or 5D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = params['padding'] ?? 0;
            const dilation = 1; // dilation is not an arg for AvgPool in PyTorch
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output spatial dimensions
            const dDim = is5D ? 2 : 1;
            const hDim = is5D ? 3 : 2;
            const wDim = is5D ? 4 : 3;
            
            const D_out = poolOutputSize(inShape[dDim], kernel, stride, padding, dilation, ceil_mode);
            const H_out = poolOutputSize(inShape[hDim], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[wDim], kernel, stride, padding, dilation, ceil_mode);
            
            assert(D_out > 0 && H_out > 0 && W_out > 0, 
                `AvgPool3D output dimensions would be ${D_out}x${H_out}x${W_out}, which is invalid`);
            
            // Return appropriate shape based on input rank
            if (is5D) {
                return [inShape[0], inShape[1], D_out, H_out, W_out];
            } else {
                return [inShape[0], D_out, H_out, W_out];
            }
        },
    },

    "LPPool1D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool1d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // LPPool1D requires 3D input [N, C, L]
            assert(inShape.length === 3, `LPPool1D requires 3D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = 0; // not in signature
            const dilation = 1; // not in signature
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output length
            const L_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            
            assert(L_out > 0, `LPPool1D output length would be ${L_out}, which is invalid`);
            
            return [inShape[0], inShape[1], L_out];
        },
    },

    "LPPool2D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool2d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // LPPool2D requires 4D input [N, C, H, W]
            assert(inShape.length === 4, `LPPool2D requires 4D input tensor, got shape ${inShape}`);
            
            // Extract parameters
            const kernel = params['kernel_size'];
            const stride = params['stride'] !== undefined ? params['stride'] : kernel;
            const padding = 0; // not in signature
            const dilation = 1; // not in signature 
            const ceil_mode = params['ceil_mode'] ?? false;
            
            // Calculate output spatial dimensions
            const H_out = poolOutputSize(inShape[2], kernel, stride, padding, dilation, ceil_mode);
            const W_out = poolOutputSize(inShape[3], kernel, stride, padding, dilation, ceil_mode);
            
            assert(H_out > 0 && W_out > 0, `LPPool2D output dimensions would be ${H_out}x${W_out}, which is invalid`);
            
            return [inShape[0], inShape[1], H_out, W_out];
        },
    },

    // Adaptive pooling
    "AdaptiveAvgPool1D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool1d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveAvgPool1D requires 3D input [N, C, L]
            assert(inShape.length === 3, `AdaptiveAvgPool1D requires 3D input tensor, got shape ${inShape}`);
            
            // Ensure output_size is valid
            const output_size = params['output_size'];
            assert(output_size > 0, `AdaptiveAvgPool1D output_size must be > 0, got ${output_size}`);
            
            return [inShape[0], inShape[1], output_size];
        },
    },

    "AdaptiveAvgPool2D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool2d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveAvgPool2D requires 4D input [N, C, H, W]
            assert(inShape.length === 4, `AdaptiveAvgPool2D requires 4D input tensor, got shape ${inShape}`);
            
            // Get output size
            const outSize = ensureArray(params['output_size']);
            
            // Validate output size
            if (outSize.length === 1) {
                assert(outSize[0] > 0, `AdaptiveAvgPool2D output_size must be > 0, got ${outSize[0]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[0]]; // Square output
            } else {
                assert(outSize.length === 2, `AdaptiveAvgPool2D expected output_size to have 1 or 2 dimensions, got ${outSize.length}`);
                assert(outSize[0] > 0 && outSize[1] > 0, 
                    `AdaptiveAvgPool2D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[1]];
            }
        },
    },

    "AdaptiveAvgPool3D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool3d(output_size=${params['output_size']})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveAvgPool3D requires 5D input [N, C, D, H, W]
            assert(inShape.length === 5, `AdaptiveAvgPool3D requires 5D input tensor, got shape ${inShape}`);
            
            // Get output size
            const outSize = ensureArray(params['output_size']);
            
            // Validate and handle different output size formats
            if (outSize.length === 1) {
                assert(outSize[0] > 0, `AdaptiveAvgPool3D output_size must be > 0, got ${outSize[0]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[0], outSize[0]]; // Cube output
            } else if (outSize.length === 2) {
                assert(outSize[0] > 0 && outSize[1] > 0, 
                    `AdaptiveAvgPool3D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[1]];
            } else {
                assert(outSize.length === 3, `AdaptiveAvgPool3D expected output_size to have 1, 2 or 3 dimensions, got ${outSize.length}`);
                assert(outSize[0] > 0 && outSize[1] > 0 && outSize[2] > 0, 
                    `AdaptiveAvgPool3D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}x${outSize[2]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[2]];
            }
        },
    },

    "AdaptiveMaxPool1D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool1d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveMaxPool1D requires 3D input [N, C, L]
            assert(inShape.length === 3, `AdaptiveMaxPool1D requires 3D input tensor, got shape ${inShape}`);
            
            // Ensure output_size is valid
            const output_size = params['output_size'];
            assert(output_size > 0, `AdaptiveMaxPool1D output_size must be > 0, got ${output_size}`);
            
            return [inShape[0], inShape[1], output_size];
        },
    },

    "AdaptiveMaxPool2D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool2d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveMaxPool2D requires 4D input [N, C, H, W]
            assert(inShape.length === 4, `AdaptiveMaxPool2D requires 4D input tensor, got shape ${inShape}`);
            
            // Get output size
            const outSize = ensureArray(params['output_size']);
            
            // Validate output size
            if (outSize.length === 1) {
                assert(outSize[0] > 0, `AdaptiveMaxPool2D output_size must be > 0, got ${outSize[0]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[0]]; // Square output
            } else {
                assert(outSize.length === 2, `AdaptiveMaxPool2D expected output_size to have 1 or 2 dimensions, got ${outSize.length}`);
                assert(outSize[0] > 0 && outSize[1] > 0, 
                    `AdaptiveMaxPool2D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[1]];
            }
        },
    },

    "AdaptiveMaxPool3D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool3d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`,
        forward_shape_inference: (inShape, params) => {
            // AdaptiveMaxPool3D requires 5D input [N, C, D, H, W]
            assert(inShape.length === 5, `AdaptiveMaxPool3D requires 5D input tensor, got shape ${inShape}`);
            
            // Get output size
            const outSize = ensureArray(params['output_size']);
            
            // Validate and handle different output size formats
            if (outSize.length === 1) {
                assert(outSize[0] > 0, `AdaptiveMaxPool3D output_size must be > 0, got ${outSize[0]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[0], outSize[0]]; // Cube output
            } else if (outSize.length === 2) {
                assert(outSize[0] > 0 && outSize[1] > 0, 
                    `AdaptiveMaxPool3D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}`);
                return [inShape[0], inShape[1], outSize[0], outSize[1], outSize[1]];
            } else {
                assert(outSize.length === 3, `AdaptiveMaxPool3D expected output_size to have 1, 2 or 3 dimensions, got ${outSize.length}`);
                assert(outSize[0] > 0 && outSize[1] > 0 && outSize[2] > 0, 
                    `AdaptiveMaxPool3D output dimensions must be > 0, got ${outSize[0]}x${outSize[1]}x${outSize[2]}`);
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
            // BatchNorm1D requires 2D [N, C] or 3D [N, C, L] input
            if (inShape.length !== 2 && inShape.length !== 3) {
                throw new Error(`BatchNorm1D requires 2D or 3D input tensor, got shape ${inShape}`);
            }
            
            // BatchNorm1D preserves input shape
            return [...inShape];
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
        forward_shape_inference: (inShape, params) => {
            // BatchNorm2D requires 3D [C, H, W] or 4D [N, C, H, W] input
            const is4D = inShape.length === 4;
            const is3D = inShape.length === 3;
            
            if (!is4D && !is3D) {
                throw new Error(`BatchNorm2D requires 3D or 4D input tensor, got shape ${inShape}`);
            }
            
            // Validate channels
            const channelDim = is4D ? 1 : 0;
            if (inShape[channelDim] !== params['num_features']) {
                throw new Error(`BatchNorm2D expected num_features=${params['num_features']}, got ${inShape[channelDim]}`);
            }
            
            // BatchNorm2D preserves input shape
            return [...inShape];
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
        forward_shape_inference: (inShape, params) => {
            // BatchNorm3D requires 4D [C, D, H, W] or 5D [N, C, D, H, W] input
            const is5D = inShape.length === 5;
            const is4D = inShape.length === 4;
            
            if (!is5D && !is4D) {
                throw new Error(`BatchNorm3D requires 4D or 5D input tensor, got shape ${inShape}`);
            }
            
            // Validate channels
            const channelDim = is5D ? 1 : 0;
            if (inShape[channelDim] !== params['num_features']) {
                throw new Error(`BatchNorm3D expected num_features=${params['num_features']}, got ${inShape[channelDim]}`);
            }
            
            // BatchNorm3D preserves input shape
            return [...inShape];
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
        forward_shape_inference: (inShape, params) => {
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
        forward_shape_inference: (inShape, params) => inShape,
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
        forward_shape_inference: (inShape, params) => {
            // Dropout takes any shape input
            assert(inShape.length > 0, `Dropout requires at least 1D input tensor, got shape ${inShape}`);
            
            // If dim is specified, validate it's within bounds
            if (params['dim'] !== undefined) {
                const dim = params['dim'];
                assert(dim >= 0 && dim < inShape.length, `Dropout dimension ${dim} is out of bounds for input shape ${inShape}`);
            }
            
            // Dropout preserves input shape
            return [...inShape];
        },
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
        forward_shape_inference: (inShape, params) => {
            // Dropout2d typically expects 4D input [N, C, H, W]
            assert(inShape.length === 4, `Dropout2D typically expects 4D input tensor, got shape ${inShape}`);
            
            // If dim is specified, validate it's within bounds
            if (params['dim'] !== undefined) {
                const dim = params['dim'];
                assert(dim >= 0 && dim < inShape.length, `Dropout2D dimension ${dim} is out of bounds for input shape ${inShape}`);
            }
            
            // Dropout2D preserves input shape
            return [...inShape];
        },
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
        forward_shape_inference: (inShape, params) => {
            // Dropout3d typically expects 5D input [N, C, D, H, W]
            assert(inShape.length === 5, `Dropout3D typically expects 5D input tensor, got shape ${inShape}`);
            
            // If dim is specified, validate it's within bounds
            if (params['dim'] !== undefined) {
                const dim = params['dim'];
                assert(dim >= 0 && dim < inShape.length, `Dropout3D dimension ${dim} is out of bounds for input shape ${inShape}`);
            }
            
            // Dropout3D preserves input shape
            return [...inShape];
        },
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
        optional_params: ["lambda"],
        code_generator: (params) => `nn.Hardshrink(lambd=${params['lambda'] ?? 0.5})`,
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
        optional_params: ["lambda"],
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
            // Validate the shape parameter is an array
            const targetShape = params['shape'];
            assert(Array.isArray(targetShape), `Reshape shape parameter must be an array, got ${targetShape}`);
            
            // Check that the number of elements is preserved (if no -1 is present)
            const hasNegOne = targetShape.includes(-1);
            
            if (!hasNegOne) {
                // Calculate total elements in input and target shapes
                const inElements = inShape.reduce((acc, dim) => acc * dim, 1);
                const outElements = targetShape.reduce((acc, dim) => acc * dim, 1);
                
                assert(inElements === outElements, 
                    `Reshape total elements mismatch: input has ${inElements} elements, but target shape has ${outElements} elements`);
            } else {
                // Count negative ones
                const negOnes = targetShape.filter(d => d === -1).length;
                assert(negOnes === 1, `Reshape shape can have at most one -1 dimension, got ${negOnes}`);
                
                // Calculate the value for the -1 dimension
                const inElements = inShape.reduce((acc, dim) => acc * dim, 1);
                const specifiedElements = targetShape.filter(d => d !== -1).reduce((acc, dim) => acc * dim, 1);
                
                assert(inElements % specifiedElements === 0, 
                    `Reshape cannot infer size for -1 dimension: input has ${inElements} elements, which is not divisible by product of specified dimensions (${specifiedElements})`);
            }
            
            // Return the target shape, as validation passed
            return targetShape;
        },
    },

    "Permute": {
        required_params: ["dims"],
        optional_params: [],
        code_generator: (params) => `torch.permute(${params['dims'].join(', ')})`,
        forward_shape_inference: (inShape, params) => {
            // Validate the dims parameter
            const dims = params['dims'];
            assert(Array.isArray(dims), `Permute dims parameter must be an array, got ${dims}`);
            assert(dims.length === inShape.length, 
                `Permute dims must have same length as input shape, got ${dims.length} vs ${inShape.length}`);
            
            // Check that all dimensions are present exactly once
            const sorted = [...dims].sort((a, b) => a - b);
            for (let i = 0; i < sorted.length; i++) {
                assert(sorted[i] === i, `Permute dims must contain all dimensions from 0 to ${inShape.length - 1} exactly once`);
            }
            
            // Calculate output shape
            const outShape = dims.map(d => inShape[d]);
            return outShape;
        },
    },

    "Flatten": {
        required_params: [],
        optional_params: ["start_dim", "end_dim"],
        code_generator: (params) => `nn.Flatten(start_dim=${params['start_dim'] ?? 1}, end_dim=${params['end_dim'] ?? -1})`,
        forward_shape_inference: (inShape, params) => {
            // Validate parameters with defaults
            const start_dim = params['start_dim'] ?? 1;
            let end_dim = params['end_dim'] ?? -1;
            
            // Convert negative end_dim to positive index
            if (end_dim < 0) {
                end_dim = inShape.length + end_dim;
            }
            
            // Validate start_dim and end_dim
            if (start_dim < 0 || start_dim >= inShape.length) {
                throw new Error(`Flatten invalid start_dim=${start_dim} for input shape ${inShape}`);
            }
            
            if (end_dim < start_dim || end_dim >= inShape.length) {
                throw new Error(`Flatten invalid end_dim=${end_dim} for input shape ${inShape}`);
            }
            
            // Calculate flattened size
            let flattenedSize = 1;
            for (let i = start_dim; i <= end_dim; i++) {
                flattenedSize *= inShape[i];
            }
            
            // Create output shape
            const outShape = [
                ...inShape.slice(0, start_dim),
                flattenedSize,
                ...inShape.slice(end_dim + 1)
            ];
            
            return outShape;
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
