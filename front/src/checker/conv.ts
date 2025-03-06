import { Shape } from './shape';
import { CheckerNode, NodeParams } from './node';
import { NodeMetadata } from './node';

interface ConvParams extends NodeParams {
    in_channels: number;
    out_channels: number;
    kernel_size: Shape;
    stride?: Shape;
    padding?: Shape | 'valid' | 'same';
    dilation?: Shape;
    groups?: number;
}

abstract class ConvBase extends CheckerNode<ConvParams> {
    protected abstract readonly dims: number;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            label: 'Convolution',
            description: 'Base convolution layer',
            category: 'convolution',
            paramFields: {
                in_channels: {
                    label: 'Input Channels',
                    description: 'Number of input channels',
                    type: 'number'
                },
                out_channels: {
                    label: 'Output Channels',
                    description: 'Number of output channels',
                    type: 'number'
                },
                kernel_size: {
                    label: 'Kernel Size',
                    description: 'Size of the convolving kernel',
                    type: 'shape'
                },
                stride: {
                    label: 'Stride',
                    description: 'Stride of the convolution',
                    type: 'shape'
                },
                padding: {
                    label: 'Padding',
                    description: 'Padding added to both sides of the input',
                    type: 'option',
                    options: ['valid', 'same'] as const
                },
                dilation: {
                    label: 'Dilation',
                    description: 'Spacing between kernel elements',
                    type: 'shape'
                },
                groups: {
                    label: 'Groups',
                    description: 'Number of blocked connections from input to output channels',
                    type: 'number'
                }
            }
        } as const;
    }

    validate_params(): void {
        const { in_channels, out_channels, kernel_size, stride, padding, dilation, groups } = this.params;
        
        if (in_channels <= 0) throw new Error('in_channels must be positive');
        if (out_channels <= 0) throw new Error('out_channels must be positive');
        if (kernel_size.length !== this.dims) throw new Error(`kernel_size must have ${this.dims} dimensions`);
        if (stride && stride.length !== this.dims) throw new Error(`stride must have ${this.dims} dimensions`);
        if (padding && typeof padding !== 'string' && padding.length !== this.dims) throw new Error(`padding must have ${this.dims} dimensions`);
        if (dilation && dilation.length !== this.dims) throw new Error(`dilation must have ${this.dims} dimensions`);
        if (groups && (groups <= 0 || in_channels % groups !== 0 || out_channels % groups !== 0)) {
            throw new Error('invalid groups configuration');
        }
    }

    compute_out_shape(in_shape: Shape): Shape {
        if (in_shape.length !== this.dims + 2) {
            throw new Error(`Input must have ${this.dims + 2} dimensions (batch, channels, spatial...)`);
        }
        if (in_shape[1] !== this.params.in_channels) {
            throw new Error(`Input has ${in_shape[1]} channels but expected ${this.params.in_channels}`);
        }

        const { kernel_size, stride = new Shape(Array(this.dims).fill(1)), 
               padding = new Shape(Array(this.dims).fill(0)), 
               dilation = new Shape(Array(this.dims).fill(1)) } = this.params;

        const pad = typeof padding === 'string' ? new Shape(Array(this.dims).fill(0)) : padding;
        
        // Output shape for each spatial dimension: 
        // out = floor((in + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
        const spatial = in_shape.slice(2).map((size, i) => {
            const dilated = dilation[i] * (kernel_size[i] - 1) + 1;
            return Math.floor((size + 2 * pad[i] - dilated - 1) / stride[i] + 1);
        });

        return new Shape([in_shape[0], this.params.out_channels, ...spatial]);
    }
}

export class Conv1d extends ConvBase {
    static readonly type = 'conv1d' as const;
    static readonly description = 'Applies a 1D convolution over an input signal composed of several input planes.';
    protected readonly dims = 1;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'Conv1D',
            description: this.description,
        };
    }
}

export class Conv2d extends ConvBase {
    static readonly type = 'conv2d' as const;
    static readonly description = 'Applies a 2D convolution over an input signal composed of several input planes.';
    protected readonly dims = 2;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'Conv2D',
            description: this.description,
        };
    }
}

export class Conv3d extends ConvBase {
    static readonly type = 'conv3d' as const;
    static readonly description = 'Applies a 3D convolution over an input signal composed of several input planes.';
    protected readonly dims = 3;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'Conv3D',
            description: this.description,
        };
    }
}

abstract class ConvTransposeBase extends ConvBase {
    compute_out_shape(in_shape: Shape): Shape {
        if (in_shape.length !== this.dims + 2) {
            throw new Error(`Input must have ${this.dims + 2} dimensions (batch, channels, spatial...)`);
        }
        if (in_shape[1] !== this.params.in_channels) {
            throw new Error(`Input has ${in_shape[1]} channels but expected ${this.params.in_channels}`);
        }

        const { kernel_size, stride = new Shape(Array(this.dims).fill(1)), 
               padding = new Shape(Array(this.dims).fill(0)), 
               dilation = new Shape(Array(this.dims).fill(1)) } = this.params;

        const pad = typeof padding === 'string' ? new Shape(Array(this.dims).fill(0)) : padding;
        
        // Output shape for each spatial dimension (transposed): 
        // out = (in - 1)*stride - 2*pad + dilation*(kernel-1) + 1
        const spatial = in_shape.slice(2).map((size, i) => {
            const dilated = dilation[i] * (kernel_size[i] - 1) + 1;
            return (size - 1) * stride[i] - 2 * pad[i] + dilated;
        });

        return new Shape([in_shape[0], this.params.out_channels, ...spatial]);
    }
}

export class ConvTranspose1d extends ConvTransposeBase {
    static readonly type = 'conv_transpose1d' as const;
    static readonly description = 'Applies a 1D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 1;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'ConvTranspose1D',
            description: this.description,
        };
    }
}

export class ConvTranspose2d extends ConvTransposeBase {
    static readonly type = 'conv_transpose2d' as const;
    static readonly description = 'Applies a 2D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 2;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'ConvTranspose2D',
            description: this.description,
        };
    }
}

export class ConvTranspose3d extends ConvTransposeBase {
    static readonly type = 'conv_transpose3d' as const;
    static readonly description = 'Applies a 3D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 3;

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(),
            label: 'ConvTranspose3D',
            description: this.description,
        };
    }
}

// Export types for the registry
export const ConvNodes = {
    [Conv1d.type]: Conv1d,
    [Conv2d.type]: Conv2d,
    [Conv3d.type]: Conv3d,
    [ConvTranspose1d.type]: ConvTranspose1d,
    [ConvTranspose2d.type]: ConvTranspose2d,
    [ConvTranspose3d.type]: ConvTranspose3d,
} as const;

export type ConvNodeParams = {
    [Conv1d.type]: ConvParams;
    [Conv2d.type]: ConvParams;
    [Conv3d.type]: ConvParams;
    [ConvTranspose1d.type]: ConvParams;
    [ConvTranspose2d.type]: ConvParams;
    [ConvTranspose3d.type]: ConvParams;
};
