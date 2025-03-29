import { Shape } from './shape';
import { CheckerNode, NodeParams, OutputError } from './node';
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

export function validateConvParams(params: Partial<ConvParams>, dims: number): string | null {
    const { in_channels, out_channels, kernel_size, stride, padding, dilation, groups } = params;
    
    if (!in_channels || in_channels <= 0) return 'Input channels must be positive';
    if (!out_channels || out_channels <= 0) return 'Output channels must be positive';
    if (!kernel_size) return 'Kernel size must be specified';
    if (kernel_size.length !== dims) return `Kernel size must be ${dims}D (got ${kernel_size.length}D: ${kernel_size})`;
    if (stride && stride.length !== dims) return `Stride must be ${dims}D (got ${stride.length}D: ${stride})`;
    if (padding && typeof padding !== 'string' && padding.length !== dims) return `Padding must be ${dims}D (got ${padding.length}D: ${padding})`;
    if (dilation && dilation.length !== dims) return `Dilation must be ${dims}D (got ${dilation.length}D: ${dilation})`;
    if (groups && (groups <= 0 || in_channels % groups !== 0 || out_channels % groups !== 0)) {
        return 'Invalid groups configuration';
    }
    return null;
}

abstract class ConvBase extends CheckerNode<ConvParams> {
    protected abstract readonly dims: number;

    static getMeta(dims: number): NodeMetadata<ConvParams> {
        return {
            label: 'Convolution',
            description: 'Base convolution layer',
            category: 'convolution',
            paramFields: {
                in_channels: {
                    label: 'Input Channels',
                    description: 'Number of input channels',
                    type: 'number',
                    default: 64
                },
                out_channels: {
                    label: 'Output Channels',
                    description: 'Number of output channels',
                    type: 'number',
                    default: 64
                },
                kernel_size: {
                    label: 'Kernel Size',
                    description: 'Size of the convolving kernel',
                    type: 'shape',
                    default: Array(dims).fill(3)
                },
                stride: {
                    label: 'Stride',
                    description: 'Stride of the convolution',
                    type: 'shape',
                    default: Array(dims).fill(1)
                },
                padding: {
                    label: 'Padding',
                    description: 'Padding added to both sides of the input',
                    type: 'option',
                    options: ['valid', 'same'] as const,
                    default: 'valid'
                },
                dilation: {
                    label: 'Dilation',
                    description: 'Spacing between kernel elements',
                    type: 'shape',
                    default: Array(dims).fill(1)
                },
                groups: {
                    label: 'Groups',
                    description: 'Number of blocked connections from input to output channels',
                    type: 'number',
                    default: 1
                }
            }
        } as const;
    }

    // Helper function for common conv parameter validation
    protected static validateConvParams(params: NodeParams, dims: number): string | null {
        if (!params || typeof params !== 'object') return 'Invalid params';
        const p = params as ConvParams;
        
        if (!p.in_channels || typeof p.in_channels !== 'number') return 'Input channels must be specified as a number';
        if (p.in_channels <= 0) return 'Input channels must be positive';
        
        if (!p.out_channels || typeof p.out_channels !== 'number') return 'Output channels must be specified as a number';
        if (p.out_channels <= 0) return 'Output channels must be positive';
        
        if (!p.kernel_size) return 'Kernel size must be specified';
        if (!Array.isArray(p.kernel_size)) return 'Kernel size must be an array';
        if (p.kernel_size.length !== dims) return `Kernel size must be ${dims}D (got ${p.kernel_size.length}D: ${p.kernel_size})`;
        
        if (p.stride) {
            if (!Array.isArray(p.stride)) return 'Stride must be an array';
            if (p.stride.length !== dims) return `Stride must be ${dims}D (got ${p.stride.length}D: ${p.stride})`;
        }
        
        if (p.padding) {
            if (typeof p.padding === 'string') {
                if (!['valid', 'same'].includes(p.padding)) return 'Padding must be "valid" or "same" if specified as string';
            } else {
                if (!Array.isArray(p.padding)) return 'Padding must be an array if not "valid" or "same"';
                if (p.padding.length !== dims) return `Padding must be ${dims}D (got ${p.padding.length}D: ${p.padding})`;
            }
        }
        
        if (p.dilation) {
            if (!Array.isArray(p.dilation)) return 'Dilation must be an array';
            if (p.dilation.length !== dims) return `Dilation must be ${dims}D (got ${p.dilation.length}D: ${p.dilation})`;
        }
        
        if (p.groups !== undefined) {
            if (typeof p.groups !== 'number') return 'Groups must be a number';
            if (p.groups <= 0) return 'Groups must be positive';
            if (p.in_channels % p.groups !== 0) return 'Input channels must be divisible by groups';
            if (p.out_channels % p.groups !== 0) return 'Output channels must be divisible by groups';
        }
        
        return null;
    }

    computeOutShape(in_shape: Shape): Shape {
        try {
            Shape.validateSpatialDims(in_shape, this.dims);
            Shape.validateChannels(in_shape, this.params.in_channels);

            const { kernel_size, stride = Array(this.dims).fill(1), 
                   padding = 'valid', 
                   dilation = Array(this.dims).fill(1) } = this.params;

            let pad: Shape;
            if (typeof padding === 'string') {
                if (padding === 'valid') {
                    pad = Array(this.dims).fill(0);
                } else if (padding === 'same') {
                    // For each dimension, calculate the padding needed
                    pad = Array(this.dims).fill(0).map((_, i) => {
                        const dilatedKernelSize = dilation[i] * (kernel_size[i] - 1) + 1;
                        const strideVal = stride[i];
                        const inputSize = in_shape[i + 2];
                        const outputSize = Math.ceil(inputSize / strideVal);
                        // Solve for pad:
                        // pad = ((outputSize - 1)*stride + dilatedKernelSize - inputSize)/2
                        const totalPad = (outputSize - 1) * strideVal + dilatedKernelSize - inputSize;
                        return Math.floor(totalPad / 2);  // Round down for asymmetric padding if needed
                    });
                } else {
                    throw new OutputError(`Invalid padding mode: ${padding}`);
                }
            } else {
                pad = padding;
            }
            
            // Output shape for each spatial dimension: 
            // out = floor((in + 2*pad - dilation*(kernel-1) - 1)/stride + 1)
            const spatial = in_shape.slice(2).map((size, i) => {
                const dilated = dilation[i] * (kernel_size[i] - 1) + 1;
                return Math.floor((size + 2 * pad[i] - dilated - 1) / stride[i] + 1);
            });

            return [in_shape[0], this.params.out_channels, ...spatial];
        } catch (e) {
            throw new OutputError(e instanceof Error ? e.message : String(e));
        }
    }
}

export class Conv1d extends ConvBase {
    static readonly type = 'conv1d' as const;
    static readonly description = 'Applies a 1D convolution over an input signal composed of several input planes.';
    protected readonly dims = 1;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 1);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(1),
            label: 'Conv1D',
            description: this.description,
        };
    }
}

export class Conv2d extends ConvBase {
    static readonly type = 'conv2d' as const;
    static readonly description = 'Applies a 2D convolution over an input signal composed of several input planes.';
    protected readonly dims = 2;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 2);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(2),
            label: 'Conv2D',
            description: this.description,
        };
    }
}

export class Conv3d extends ConvBase {
    static readonly type = 'conv3d' as const;
    static readonly description = 'Applies a 3D convolution over an input signal composed of several input planes.';
    protected readonly dims = 3;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 3);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(3),
            label: 'Conv3D',
            description: this.description,
        };
    }
}

abstract class ConvTransposeBase extends ConvBase {
    computeOutShape(in_shape: Shape): Shape {
        try {
            Shape.validateSpatialDims(in_shape, this.dims);
            Shape.validateChannels(in_shape, this.params.in_channels);

            const { kernel_size, stride = Array(this.dims).fill(1), 
                   padding = 'valid', 
                   dilation = Array(this.dims).fill(1) } = this.params;

            let pad: Shape;
            if (typeof padding === 'string') {
                if (padding === 'valid') {
                    pad = Array(this.dims).fill(0);
                } else if (padding === 'same') {
                    // For transposed convolutions with 'same' padding, we want:
                    // output_size = input_size * stride
                    // This means we need to adjust padding to achieve this size
                    pad = Array(this.dims).fill(0).map((_, i) => {
                        const dilatedKernelSize = dilation[i] * (kernel_size[i] - 1) + 1;
                        const strideVal = stride[i];
                        const inputSize = in_shape[i + 2];
                        const outputSize = inputSize * strideVal;
                        // For transposed conv, we need:
                        // outputSize = (inputSize - 1)*stride - 2*pad + dilatedKernelSize
                        // Solve for pad:
                        const totalPad = ((inputSize - 1) * strideVal + dilatedKernelSize - outputSize) / 2;
                        return Math.floor(totalPad);  // Round down for asymmetric padding if needed
                    });
                } else {
                    throw new OutputError(`Invalid padding mode: ${padding}`);
                }
            } else {
                pad = padding;
            }
            
            // Output shape for each spatial dimension (transposed): 
            // out = (in - 1)*stride - 2*pad + dilation*(kernel-1) + 1
            const spatial = in_shape.slice(2).map((size, i) => {
                const dilated = dilation[i] * (kernel_size[i] - 1) + 1;
                return (size - 1) * stride[i] - 2 * pad[i] + dilated;
            });

            return [in_shape[0], this.params.out_channels, ...spatial];
        } catch (e) {
            throw new OutputError(e instanceof Error ? e.message : String(e));
        }
    }
}

export class ConvTranspose1d extends ConvTransposeBase {
    static readonly type = 'conv_transpose1d' as const;
    static readonly description = 'Applies a 1D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 1;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 1);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(1),
            label: 'ConvTranspose1D',
            description: this.description,
        };
    }
}

export class ConvTranspose2d extends ConvTransposeBase {
    static readonly type = 'conv_transpose2d' as const;
    static readonly description = 'Applies a 2D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 2;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 2);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(2),
            label: 'ConvTranspose2D',
            description: this.description,
        };
    }
}

export class ConvTranspose3d extends ConvTransposeBase {
    static readonly type = 'conv_transpose3d' as const;
    static readonly description = 'Applies a 3D transposed convolution operator over an input image composed of several input planes.';
    protected readonly dims = 3;

    static validateParams(params: NodeParams): string | null {
        return ConvBase.validateConvParams(params, 3);
    }

    static getMeta(): NodeMetadata<ConvParams> {
        return {
            ...super.getMeta(3),
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
};

export type ConvNodeParams = {
    [Conv1d.type]: ConvParams;
    [Conv2d.type]: ConvParams;
    [Conv3d.type]: ConvParams;
    [ConvTranspose1d.type]: ConvParams;
    [ConvTranspose2d.type]: ConvParams;
    [ConvTranspose3d.type]: ConvParams;
}
