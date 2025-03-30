/**
 * This file defines Flax modules, their parameters, and shape inference functions.
 * It focuses on modules that take a single input and produce a single output.
 */

interface FlaxModuleMetadata {
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
    padding: "SAME" | "VALID" | number
): number {
    if (padding === "SAME") {
        return Math.ceil(inSize / stride);
    } else if (padding === "VALID") {
        return Math.floor((inSize - kernelSize) / stride) + 1;
    } else {
        // If padding is a number
        return Math.floor((inSize + 2 * padding - kernelSize) / stride) + 1;
    }
}

function convTransposeOutputSize(
    inSize: number,
    kernelSize: number,
    stride: number,
    padding: "SAME" | "VALID" | number
): number {
    if (padding === "SAME") {
        return inSize * stride;
    } else if (padding === "VALID") {
        return (inSize - 1) * stride + kernelSize;
    } else {
        // If padding is a number
        return (inSize - 1) * stride - 2 * padding + kernelSize;
    }
}

function poolOutputSize(
    inSize: number,
    windowSize: number,
    stride: number,
    padding: "SAME" | "VALID" | number
): number {
    if (padding === "SAME") {
        return Math.ceil(inSize / stride);
    } else if (padding === "VALID") {
        return Math.floor((inSize - windowSize) / stride) + 1;
    } else {
        // If padding is a number
        return Math.floor((inSize + 2 * padding - windowSize) / stride) + 1;
    }
}

function resolvePadding(padding: any): "SAME" | "VALID" | number {
    if (padding === "SAME" || padding === "VALID") {
        return padding;
    }
    
    // If it's a number, return it directly
    if (typeof padding === "number") {
        return padding;
    }
    
    // By default, use VALID padding
    return "VALID";
}

export const flax_module_metadata: Record<string, FlaxModuleMetadata> = {
    // Dense (Linear) layer
    "Dense": {
        required_params: ["features"],
        optional_params: ["use_bias", "dtype", "param_dtype", "precision", "kernel_init", "bias_init"],
        code_generator: (params) => 
            `nn.Dense(features=${params.features}, ` +
            `use_bias=${params.use_bias ?? true}, ` +
            `kernel_init=${params.kernel_init ?? 'nn.initializers.lecun_normal()'}, ` +
            `bias_init=${params.bias_init ?? 'nn.initializers.zeros'})`,
        forward_shape_inference: (inShape, params) => {
            // Dense expects shape [batch_size, ..., features_in]
            // and outputs [batch_size, ..., features]
            const outShape = [...inShape];
            outShape[outShape.length - 1] = params.features;
            return outShape;
        }
    },

    // Convolution layers
    "Conv": {
        required_params: ["features", "kernel_size"],
        optional_params: ["strides", "padding", "input_dilation", "kernel_dilation", "feature_group_count", "use_bias", "dtype", "param_dtype", "precision", "kernel_init", "bias_init"],
        code_generator: (params) => {
            const kernel_size_str = Array.isArray(params.kernel_size) ? 
                `(${params.kernel_size.join(', ')})` : `(${params.kernel_size},)`;
            const strides_str = Array.isArray(params.strides) ? 
                `(${params.strides.join(', ')})` : `(${params.strides ?? 1},)`;
            
            return `nn.Conv(features=${params.features}, ` +
                `kernel_size=${kernel_size_str}, ` +
                `strides=${strides_str}, ` +
                `padding=${typeof params.padding === 'string' ? `'${params.padding}'` : params.padding ?? "'VALID'"}, ` +
                `use_bias=${params.use_bias ?? true}, ` +
                `kernel_init=${params.kernel_init ?? 'nn.initializers.lecun_normal()'}, ` +
                `bias_init=${params.bias_init ?? 'nn.initializers.zeros'})`;
        },
        forward_shape_inference: (inShape, params) => {
            // Conv expects [batch_size, spatial_dims..., in_features]
            // and outputs [batch_size, spatial_dims..., out_features]
            const spatialDims = inShape.length - 2;
            const outShape = [inShape[0]]; // Start with batch size
            
            // Process spatial dimensions
            const kernelSize = Array.isArray(params.kernel_size) ? params.kernel_size : [params.kernel_size];
            const strides = Array.isArray(params.strides) ? params.strides : Array(spatialDims).fill(params.strides ?? 1);
            const padding = resolvePadding(params.padding ?? "VALID");
            
            for (let i = 0; i < spatialDims; i++) {
                const kSize = kernelSize.length > i ? kernelSize[i] : kernelSize[0];
                const stride = strides.length > i ? strides[i] : strides[0];
                outShape.push(convOutputSize(inShape[i + 1], kSize, stride, padding));
            }
            
            // Add output channels
            outShape.push(params.features);
            
            return outShape;
        }
    },

    // Transposed convolution
    "ConvTranspose": {
        required_params: ["features", "kernel_size"],
        optional_params: ["strides", "padding", "kernel_dilation", "use_bias", "dtype", "param_dtype", "precision", "kernel_init", "bias_init"],
        code_generator: (params) => {
            const kernel_size_str = Array.isArray(params.kernel_size) ? 
                `(${params.kernel_size.join(', ')})` : `(${params.kernel_size},)`;
            const strides_str = Array.isArray(params.strides) ? 
                `(${params.strides.join(', ')})` : `(${params.strides ?? 1},)`;
            
            return `nn.ConvTranspose(features=${params.features}, ` +
                `kernel_size=${kernel_size_str}, ` +
                `strides=${strides_str}, ` +
                `padding=${typeof params.padding === 'string' ? `'${params.padding}'` : params.padding ?? "'VALID'"}, ` +
                `use_bias=${params.use_bias ?? true}, ` +
                `kernel_init=${params.kernel_init ?? 'nn.initializers.lecun_normal()'}, ` +
                `bias_init=${params.bias_init ?? 'nn.initializers.zeros'})`;
        },
        forward_shape_inference: (inShape, params) => {
            // ConvTranspose expects [batch_size, spatial_dims..., in_features]
            // and outputs [batch_size, spatial_dims..., out_features]
            const spatialDims = inShape.length - 2;
            const outShape = [inShape[0]]; // Start with batch size
            
            // Process spatial dimensions
            const kernelSize = Array.isArray(params.kernel_size) ? params.kernel_size : [params.kernel_size];
            const strides = Array.isArray(params.strides) ? params.strides : Array(spatialDims).fill(params.strides ?? 1);
            const padding = resolvePadding(params.padding ?? "VALID");
            
            for (let i = 0; i < spatialDims; i++) {
                const kSize = kernelSize.length > i ? kernelSize[i] : kernelSize[0];
                const stride = strides.length > i ? strides[i] : strides[0];
                outShape.push(convTransposeOutputSize(inShape[i + 1], kSize, stride, padding));
            }
            
            // Add output channels
            outShape.push(params.features);
            
            return outShape;
        }
    },

    // BatchNorm
    "BatchNorm": {
        required_params: [],
        optional_params: ["use_running_average", "momentum", "epsilon", "dtype", "param_dtype", "use_bias", "use_scale", "bias_init", "scale_init", "axis", "mean_init", "var_init"],
        code_generator: (params) => 
            `nn.BatchNorm(use_running_average=${params.use_running_average ?? 'not training'}, ` +
            `momentum=${params.momentum ?? 0.9}, ` +
            `epsilon=${params.epsilon ?? 1e-5}, ` +
            `use_bias=${params.use_bias ?? true}, ` +
            `use_scale=${params.use_scale ?? true})`,
        forward_shape_inference: (inShape, params) => {
            // BatchNorm preserves shape
            return [...inShape];
        }
    },

    // LayerNorm
    "LayerNorm": {
        required_params: [],
        optional_params: ["epsilon", "dtype", "param_dtype", "use_bias", "use_scale", "bias_init", "scale_init", "reduction_axes"],
        code_generator: (params) => 
            `nn.LayerNorm(epsilon=${params.epsilon ?? 1e-5}, ` +
            `use_bias=${params.use_bias ?? true}, ` +
            `use_scale=${params.use_scale ?? true})`,
        forward_shape_inference: (inShape, params) => {
            // LayerNorm preserves shape
            return [...inShape];
        }
    },

    // Dropout
    "Dropout": {
        required_params: ["rate"],
        optional_params: ["deterministic", "broadcast_dims"],
        code_generator: (params) => 
            `nn.Dropout(rate=${params.rate}, ` +
            `deterministic=${params.deterministic ?? 'not training'})`,
        forward_shape_inference: (inShape, params) => {
            // Dropout preserves shape
            return [...inShape];
        }
    },

    // Pooling operations
    "max_pool": {
        required_params: ["window_shape"],
        optional_params: ["strides", "padding", "base_dilation", "window_dilation"],
        code_generator: (params) => {
            const window_shape_str = Array.isArray(params.window_shape) ? 
                `(${params.window_shape.join(', ')})` : `(${params.window_shape},)`;
            const strides_str = Array.isArray(params.strides) ? 
                `(${params.strides.join(', ')})` : `(${params.strides ?? params.window_shape},)`;
            
            return `nn.max_pool(x, window_shape=${window_shape_str}, ` +
                `strides=${strides_str}, ` +
                `padding=${typeof params.padding === 'string' ? `'${params.padding}'` : params.padding ?? "'VALID'"})`;
        },
        forward_shape_inference: (inShape, params) => {
            // Pooling expects [batch_size, spatial_dims..., features]
            // and preserves batch and feature dims
            const spatialDims = inShape.length - 2;
            const outShape = [inShape[0]]; // Start with batch size
            
            // Process spatial dimensions
            const windowShape = Array.isArray(params.window_shape) ? params.window_shape : [params.window_shape];
            const strides = Array.isArray(params.strides) ? 
                params.strides : 
                Array.isArray(params.window_shape) ? 
                    params.window_shape : 
                    Array(spatialDims).fill(params.strides ?? params.window_shape);
            const padding = resolvePadding(params.padding ?? "VALID");
            
            for (let i = 0; i < spatialDims; i++) {
                const wSize = windowShape.length > i ? windowShape[i] : windowShape[0];
                const stride = strides.length > i ? strides[i] : strides[0];
                outShape.push(poolOutputSize(inShape[i + 1], wSize, stride, padding));
            }
            
            // Add feature dimension
            outShape.push(inShape[inShape.length - 1]);
            
            return outShape;
        }
    },

    "avg_pool": {
        required_params: ["window_shape"],
        optional_params: ["strides", "padding", "base_dilation", "window_dilation", "count_include_pad"],
        code_generator: (params) => {
            const window_shape_str = Array.isArray(params.window_shape) ? 
                `(${params.window_shape.join(', ')})` : `(${params.window_shape},)`;
            const strides_str = Array.isArray(params.strides) ? 
                `(${params.strides.join(', ')})` : `(${params.strides ?? params.window_shape},)`;
            
            return `nn.avg_pool(x, window_shape=${window_shape_str}, ` +
                `strides=${strides_str}, ` +
                `padding=${typeof params.padding === 'string' ? `'${params.padding}'` : params.padding ?? "'VALID'"})`;
        },
        forward_shape_inference: (inShape, params) => {
            // Same shape inference as max_pool
            const spatialDims = inShape.length - 2;
            const outShape = [inShape[0]]; // Start with batch size
            
            // Process spatial dimensions
            const windowShape = Array.isArray(params.window_shape) ? params.window_shape : [params.window_shape];
            const strides = Array.isArray(params.strides) ? 
                params.strides : 
                Array.isArray(params.window_shape) ? 
                    params.window_shape : 
                    Array(spatialDims).fill(params.strides ?? params.window_shape);
            const padding = resolvePadding(params.padding ?? "VALID");
            
            for (let i = 0; i < spatialDims; i++) {
                const wSize = windowShape.length > i ? windowShape[i] : windowShape[0];
                const stride = strides.length > i ? strides[i] : strides[0];
                outShape.push(poolOutputSize(inShape[i + 1], wSize, stride, padding));
            }
            
            // Add feature dimension
            outShape.push(inShape[inShape.length - 1]);
            
            return outShape;
        }
    },

    // Activation functions - these preserve shape
    "relu": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.relu`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "gelu": {
        required_params: [],
        optional_params: ["approximate"],
        code_generator: (params) => `nn.gelu(approximate=${params.approximate ?? false})`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "sigmoid": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.sigmoid`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "tanh": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.tanh`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "silu": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.silu`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "swish": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.swish`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "leaky_relu": {
        required_params: [],
        optional_params: ["negative_slope"],
        code_generator: (params) => `nn.leaky_relu(negative_slope=${params.negative_slope ?? 0.01})`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "log_softmax": {
        required_params: [],
        optional_params: ["axis"],
        code_generator: (params) => `nn.log_softmax(axis=${params.axis ?? -1})`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    "softmax": {
        required_params: [],
        optional_params: ["axis"],
        code_generator: (params) => `nn.softmax(axis=${params.axis ?? -1})`,
        forward_shape_inference: (inShape) => [...inShape]
    },

    // Reshape and flatten operations
    "Flatten": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Flatten()`,
        forward_shape_inference: (inShape) => {
            // Flattens all but the first dimension
            const totalElements = inShape.slice(1).reduce((a, b) => a * b, 1);
            return [inShape[0], totalElements];
        }
    },
};

// Function to generate Flax code for a given module type
export function generateFlaxCode(moduleType: string, params: Record<string, any> = {}): string {
    if (!flax_module_metadata[moduleType]) {
        throw new Error(`Unknown Flax module type: ${moduleType}`);
    }
    
    return flax_module_metadata[moduleType].code_generator(params);
}

// Function to infer output shape for a given module type
export function forwardShapeInference(moduleType: string, inShape: number[], params: Record<string, any> = {}): number[] {
    if (!flax_module_metadata[moduleType]) {
        throw new Error(`No shape inference available for Flax module type: ${moduleType}`);
    }
    
    return flax_module_metadata[moduleType].forward_shape_inference(inShape, params);
}

// Function to get Flax elementwise operation code
export function getFlaxElementwiseOpCode(opType: string): string {
    switch (opType.toLowerCase()) {
        // Math operations
        case 'add':
            return 'jnp.add';
        case 'sub':
            return 'jnp.subtract';
        case 'mul':
            return 'jnp.multiply';
        case 'div':
            return 'jnp.divide';
        case 'pow':
            return 'jnp.power';
        case 'abs':
            return 'jnp.abs';
        case 'sqrt':
            return 'jnp.sqrt';
        case 'square':
            return 'jnp.square';
        case 'exp':
            return 'jnp.exp';
        case 'log':
            return 'jnp.log';
        case 'sin':
            return 'jnp.sin';
        case 'cos':
            return 'jnp.cos';
        case 'tan':
            return 'jnp.tan';
        case 'asin':
            return 'jnp.arcsin';
        case 'acos':
            return 'jnp.arccos';
        case 'atan':
            return 'jnp.arctan';
        
        // Min/max operations
        case 'min':
            return 'jnp.minimum';
        case 'max':
            return 'jnp.maximum';
        
        // Logical operations
        case 'and':
            return 'jnp.logical_and';
        case 'or':
            return 'jnp.logical_or';
        case 'xor':
            return 'jnp.logical_xor';
        case 'not':
            return 'jnp.logical_not';
        
        // Comparison operations
        case 'eq':
            return 'jnp.equal';
        case 'ne':
            return 'jnp.not_equal';
        case 'lt':
            return 'jnp.less';
        case 'le':
            return 'jnp.less_equal';
        case 'gt':
            return 'jnp.greater';
        case 'ge':
            return 'jnp.greater_equal';
            
        // Identity function
        case 'identity':
            return 'lambda x: x';
            
        default:
            throw new Error(`Unknown Flax elementwise operation: ${opType}`);
    }
} 