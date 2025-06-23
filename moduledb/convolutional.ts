import { ModuleDef } from './types';
import { NodeType } from '../OpCompiler/types';

// Helper functions for shape inference (copied from torch_nn_module_op.ts)
function convOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number
): number {
    const out = Math.floor((inSize + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1);
    if (!Number.isInteger(out) || out <= 0)
        throw new Error("Convolution output size isn't a positive integer");
    return out;
}

function convTransposeOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: number,
    dilation: number,
    outputPadding: number
): number {
    const out = (inSize - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1;
    if (!Number.isInteger(out) || out <= 0)
        throw new Error("Convolution output size isn't a positive integer");
    return out;
}

// Common parameters shared by all convolution layers
const commonConvParams = {
    in_channels: {
        label: 'Input Channels',
        description: 'Number of channels in the input',
        type: 'number' as const,
        required: true
    },
    out_channels: {
        label: 'Output Channels',
        description: 'Number of channels produced by the convolution',
        type: 'number' as const,
        required: true
    },
    kernel_size: {
        label: 'Kernel Size',
        description: 'Size of the convolving kernel',
        type: 'number' as const,
        required: true
    },
    stride: {
        label: 'Stride',
        description: 'Stride of the convolution',
        type: 'number' as const,
        default: 1,
        required: false
    },
    padding: {
        label: 'Padding',
        description: 'Padding added to the input',
        type: 'number' as const,
        default: 0,
        required: false
    },
    dilation: {
        label: 'Dilation',
        description: 'Spacing between kernel elements',
        type: 'number' as const,
        default: 1,
        required: false
    },
    groups: {
        label: 'Groups',
        description: 'Number of blocked connections from input to output channels',
        type: 'number' as const,
        default: 1,
        required: false
    },
    bias: {
        label: 'Bias',
        description: 'Whether to add a learnable bias to the output',
        type: 'boolean' as const,
        default: true,
        required: false
    },
    padding_mode: {
        label: 'Padding Mode',
        description: 'Type of padding algorithm',
        type: 'option' as const,
        options: ['zeros', 'reflect', 'replicate', 'circular'],
        default: 'zeros',
        required: false
    }
};

// Additional parameters for transposed convolutions
const transposeConvParams = {
    output_padding: {
        label: 'Output Padding',
        description: 'Additional size added to one side of each dimension in the output shape',
        type: 'number' as const,
        default: 0,
        required: false
    }
};

export const Conv1D: ModuleDef = {
    label: 'Conv1D',
    description: 'Applies a 1D convolution over an input signal',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to both sides of the input'
        }
    },
    emitPytorchModule: (params) => 
        `nn.Conv1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
        `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 2 && inShape.length !== 3) {
            throw new Error(`Conv1D requires 2D or 3D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 3 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`Conv1D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is3D = inShape.length === 3;
        const lenDim = is3D ? 2 : 1;
        const L_out = convOutputSize(inShape[lenDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        
        if (is3D) {
            return [inShape[0], params['out_channels'], L_out];
        } else {
            return [params['out_channels'], L_out];
        }
    }
};

export const Conv2D: ModuleDef = {
    label: 'Conv2D',
    description: 'Applies a 2D convolution over an input image',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        in_channels: {
            ...commonConvParams.in_channels,
            description: 'Number of channels in the input image'
        },
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to all four sides of the input'
        }
    },
    emitPytorchModule: (params) =>
        `nn.Conv2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
        `bias=${params['bias'] ?? 'True'}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 3 && inShape.length !== 4) {
            throw new Error(`Conv2D requires 3D or 4D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 4 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`Conv2D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is4D = inShape.length === 4;
        const hDim = is4D ? 2 : 1;
        const wDim = is4D ? 3 : 2;
        
        const H_out = convOutputSize(inShape[hDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        const W_out = convOutputSize(inShape[wDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        
        if (is4D) {
            return [inShape[0], params['out_channels'], H_out, W_out];
        } else {
            return [params['out_channels'], H_out, W_out];
        }
    }
};

export const Conv3D: ModuleDef = {
    label: 'Conv3D', 
    description: 'Applies a 3D convolution over an input volume',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        in_channels: {
            ...commonConvParams.in_channels,
            description: 'Number of channels in the input volume'
        },
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to all sides of the input'
        }
    },
    emitPytorchModule: (params) =>
        `nn.Conv3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
        `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 4 && inShape.length !== 5) {
            throw new Error(`Conv3D requires 4D or 5D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 5 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`Conv3D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is5D = inShape.length === 5;
        const dDim = is5D ? 2 : 1;
        const hDim = is5D ? 3 : 2;
        const wDim = is5D ? 4 : 3;
        
        const D_out = convOutputSize(inShape[dDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        const H_out = convOutputSize(inShape[hDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        const W_out = convOutputSize(inShape[wDim], params['kernel_size'], params['stride'], params['padding'], params['dilation']);
        
        if (is5D) {
            return [inShape[0], params['out_channels'], D_out, H_out, W_out];
        } else {
            return [params['out_channels'], D_out, H_out, W_out];
        }
    }
};

export const ConvTranspose1D: ModuleDef = {
    label: 'ConvTranspose1D',
    description: 'Applies a 1D transposed convolution operator over an input signal',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        ...transposeConvParams,
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to both sides of the input'
        }
    },
    emitPytorchModule: (params) =>
        `nn.ConvTranspose1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
        `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? 'True'}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 2 && inShape.length !== 3) {
            throw new Error(`ConvTranspose1D requires 2D or 3D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 3 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`ConvTranspose1D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is3D = inShape.length === 3;
        const lenDim = is3D ? 2 : 1;
        const L_out = convTransposeOutputSize(inShape[lenDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        
        if (is3D) {
            return [inShape[0], params['out_channels'], L_out];
        } else {
            return [params['out_channels'], L_out];
        }
    }
};

export const ConvTranspose2D: ModuleDef = {
    label: 'ConvTranspose2D',
    description: 'Applies a 2D transposed convolution operator over an input image',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        ...transposeConvParams,
        in_channels: {
            ...commonConvParams.in_channels,
            description: 'Number of channels in the input image'
        },
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to all four sides of the input'
        }
    },
    emitPytorchModule: (params) =>
        `nn.ConvTranspose2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
        `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? 'True'}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 3 && inShape.length !== 4) {
            throw new Error(`ConvTranspose2D requires 3D or 4D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 4 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`ConvTranspose2D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is4D = inShape.length === 4;
        const hDim = is4D ? 2 : 1;
        const wDim = is4D ? 3 : 2;
        
        const H_out = convTransposeOutputSize(inShape[hDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        const W_out = convTransposeOutputSize(inShape[wDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        
        if (is4D) {
            return [inShape[0], params['out_channels'], H_out, W_out];
        } else {
            return [params['out_channels'], H_out, W_out];
        }
    }
};

export const ConvTranspose3D: ModuleDef = {
    label: 'ConvTranspose3D',
    description: 'Applies a 3D transposed convolution operator over an input volume',
    category: 'Convolutional',
    moduleType: NodeType.OP,
    params: {
        ...commonConvParams,
        ...transposeConvParams,
        in_channels: {
            ...commonConvParams.in_channels,
            description: 'Number of channels in the input volume'
        },
        padding: {
            ...commonConvParams.padding,
            description: 'Padding added to all sides of the input'
        }
    },
    emitPytorchModule: (params) =>
        `nn.ConvTranspose3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
        `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
        `output_padding=${params['output_padding'] ?? 0}, dilation=${params['dilation'] ?? 1}, ` +
        `groups=${params['groups'] ?? 1}, bias=${params['bias'] ?? 'True'}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`,
    validateInputShape: (inShape, params) => {
        if (inShape.length !== 4 && inShape.length !== 5) {
            throw new Error(`ConvTranspose3D requires 4D or 5D input tensor, got shape ${inShape}`);
        }
        
        const channelDim = inShape.length === 5 ? 1 : 0;
        if (inShape[channelDim] !== params['in_channels']) {
            throw new Error(`ConvTranspose3D expected in_channels=${params['in_channels']}, got ${inShape[channelDim]}`);
        }
        
        return [];
    },
    inferOutputShape: (inShape, params) => {
        const is5D = inShape.length === 5;
        const dDim = is5D ? 2 : 1;
        const hDim = is5D ? 3 : 2;
        const wDim = is5D ? 4 : 3;
        
        const D_out = convTransposeOutputSize(inShape[dDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        const H_out = convTransposeOutputSize(inShape[hDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        const W_out = convTransposeOutputSize(inShape[wDim], params['kernel_size'], params['stride'], params['padding'], params['dilation'], params['output_padding']);
        
        if (is5D) {
            return [inShape[0], params['out_channels'], D_out, H_out, W_out];
        } else {
            return [params['out_channels'], D_out, H_out, W_out];
        }
    }
};
