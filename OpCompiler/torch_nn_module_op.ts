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
            
            // Calculate total elements in input
            const inElements = inShape.reduce((acc, dim) => acc * dim, 1);
            
            if (!hasNegOne) {
                // Calculate total elements in target shapes
                const outElements = targetShape.reduce((acc, dim) => acc * dim, 1);
                
                assert(inElements === outElements, 
                    `Reshape total elements mismatch: input has ${inElements} elements, but target shape has ${outElements} elements`);
                
                // Return the target shape, as validation passed
                return [...targetShape];
            } else {
                // Count negative ones
                const negOnes = targetShape.filter(d => d === -1).length;
                assert(negOnes === 1, `Reshape shape can have at most one -1 dimension, got ${negOnes}`);
                
                // Calculate the value for the -1 dimension
                const specifiedElements = targetShape.filter(d => d !== -1).reduce((acc, dim) => acc * dim, 1);
                
                assert(inElements % specifiedElements === 0, 
                    `Reshape cannot infer size for -1 dimension: input has ${inElements} elements, which is not divisible by product of specified dimensions (${specifiedElements})`);
                
                // Calculate the correct value for the -1 dimension
                const negOneDimValue = inElements / specifiedElements;
                
                // Create a new output shape with the calculated dimension
                const outputShape = targetShape.map(d => d === -1 ? negOneDimValue : d);
                
                return outputShape;
            }
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
