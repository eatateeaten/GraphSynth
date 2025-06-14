import { ModuleDef } from './types';

export const Conv2D: ModuleDef = {
    label: 'Conv2D',
    description: 'Applies a 2D convolution over an input signal',
    category: 'Convolutional',
    params: {
        in_channels: {
            label: 'Input Channels',
            description: 'Number of channels in the input image',
            type: 'number',
            default: 3,
            required: true
        },
        out_channels: {
            label: 'Output Channels',
            description: 'Number of channels produced by the convolution',
            type: 'number',
            default: 64,
            required: true
        },
        kernel_size: {
            label: 'Kernel Size',
            description: 'Size of the convolving kernel',
            type: 'number',
            default: 3,
            required: true
        },
        stride: {
            label: 'Stride',
            description: 'Stride of the convolution',
            type: 'number',
            default: 1,
            required: false
        },
        padding: {
            label: 'Padding',
            description: 'Padding added to all four sides of the input',
            type: 'number',
            default: 0,
            required: false
        }
        // TODO: Add remaining params (dilation, groups, bias, padding_mode)
    },
    toPytorchExpr: (params) => `nn.Conv2d(${params.in_channels}, ${params.out_channels}, ${params.kernel_size}, stride=${params.stride ?? 1}, padding=${params.padding ?? 0})`,
    shapeInference: (inShape, params) => {
        // TODO: Copy shape inference logic from torch file
        return inShape; // Placeholder
    }
};

// TODO: Add Conv1D, Conv3D, ConvTranspose2D, etc. 