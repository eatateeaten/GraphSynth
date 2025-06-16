/**
 * This file provides mapping functions to convert PyTorch operations to their JAX/Flax equivalents.
 * It specifically covers operations defined in torch_nn_module_op.ts.
 */

import { nn_module_metadata } from './torch_nn_module_op';

// Maps torch elementwise operations to JAX operations
export function getJaxElementwiseOpCode(opType: string, input?: string): string {
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
        return input ? input : 'lambda x: x';
        
    default:
        throw new Error(`Unknown elementwise operation type for JAX conversion: ${opType}`);
    }
}

// Converts PyTorch module code to Flax module code
export function getTorchToFlaxCode(module_type: string, params: Record<string, any>): string {
    // Common conversion logic for layer parameters
    const convertCommonParams = (params: Record<string, any>) => {
        // JAX/Flax uses kernel_size instead of kernel_size (same syntax but we're explicit here)
        const kernel_size = params['kernel_size'];
        // JAX/Flax typically uses strides (plural) instead of stride
        const strides = params['stride'] ?? 1;
        // JAX/Flax conv has different padding syntax
        const padding = params['padding'] ?? 0;
        const padding_str = typeof padding === 'number' ? 
            `'SAME' if ${padding} > 0 else 'VALID'` : `'SAME'`;
        
        return { kernel_size, strides, padding_str };
    };

    switch (module_type) {
    // Linear layers
    case 'Linear':
        return `nn.Dense(features=${params['output_features']}, use_bias=${params['bias'] ?? true})`;
        
        // Convolutional layers
    case 'Conv1D':
        const conv1d = convertCommonParams(params);
        return `nn.Conv(features=${params['out_channels']}, kernel_size=(${conv1d.kernel_size},), strides=(${conv1d.strides},), padding=${conv1d.padding_str}, kernel_init=nn.initializers.lecun_normal())`;
        
    case 'Conv2D':
        const conv2d = convertCommonParams(params);
        return `nn.Conv(features=${params['out_channels']}, kernel_size=(${conv2d.kernel_size}, ${conv2d.kernel_size}), strides=(${conv2d.strides}, ${conv2d.strides}), padding=${conv2d.padding_str}, kernel_init=nn.initializers.lecun_normal())`;
        
    case 'Conv3D':
        const conv3d = convertCommonParams(params);
        return `nn.Conv(features=${params['out_channels']}, kernel_size=(${conv3d.kernel_size}, ${conv3d.kernel_size}, ${conv3d.kernel_size}), strides=(${conv3d.strides}, ${conv3d.strides}, ${conv3d.strides}), padding=${conv3d.padding_str}, kernel_init=nn.initializers.lecun_normal())`;
        
        // Transposed convolutions
    case 'ConvTranspose1D':
    case 'ConvTranspose2D':
    case 'ConvTranspose3D':
        const convT = convertCommonParams(params);
        const dimensions = module_type === 'ConvTranspose1D' ? 1 : 
            (module_type === 'ConvTranspose2D' ? 2 : 3);
            
        let kernel_size_str = '';
        let strides_str = '';
            
        for (let i = 0; i < dimensions; i++) {
            kernel_size_str += `${convT.kernel_size}, `;
            strides_str += `${convT.strides}, `;
        }
            
        // Remove trailing commas
        kernel_size_str = kernel_size_str.slice(0, -2);
        strides_str = strides_str.slice(0, -2);
            
        return `nn.ConvTranspose(features=${params['out_channels']}, kernel_size=(${kernel_size_str}), strides=(${strides_str}), padding=${convT.padding_str}, kernel_init=nn.initializers.lecun_normal())`;
        
        // Pooling layers
    case 'MaxPool1D':
        return `nn.max_pool(window_shape=(${params['kernel_size']},), strides=(${params['stride'] ?? params['kernel_size']},), padding='VALID')`;
        
    case 'MaxPool2D':
        return `nn.max_pool(window_shape=(${params['kernel_size']}, ${params['kernel_size']}), strides=(${params['stride'] ?? params['kernel_size']}, ${params['stride'] ?? params['kernel_size']}), padding='VALID')`;
        
    case 'MaxPool3D':
        return `nn.max_pool(window_shape=(${params['kernel_size']}, ${params['kernel_size']}, ${params['kernel_size']}), strides=(${params['stride'] ?? params['kernel_size']}, ${params['stride'] ?? params['kernel_size']}, ${params['stride'] ?? params['kernel_size']}), padding='VALID')`;
        
        // Activation functions
    case 'ReLU':
        return `nn.relu`;
        
    case 'LeakyReLU':
        return `nn.leaky_relu(negative_slope=${params['negative_slope'] ?? 0.01})`;
        
    case 'Sigmoid':
        return `nn.sigmoid`;
        
    case 'Tanh':
        return `nn.tanh`;
        
        // Normalization
    case 'BatchNorm1D':
    case 'BatchNorm2D':
    case 'BatchNorm3D':
        return `nn.BatchNorm(use_running_average=not training, momentum=${params['momentum'] ?? 0.9}, epsilon=${params['eps'] ?? 1e-5})`;
        
        // Dropout
    case 'Dropout':
        return `nn.Dropout(rate=${params['p'] ?? 0.5}, deterministic=not training)`;
        
    default:
        throw new Error(`Unsupported module type for JAX/Flax conversion: ${module_type}`);
    }
}

// Convert a PyTorch function chain to JAX/Flax
export function torchToJaxCode(torchCode: string): string {
    try {
        // This is a simplistic approach - a real converter would need a proper parser
        // Replace torch module calls with flax equivalents
        let jaxCode = torchCode
            // Replace common torch prefixes
            .replace(/torch\./g, 'jnp.')
            .replace(/F\./g, 'nn.')
            
            // Replace common functions
            .replace(/\.view\(/g, '.reshape(')
            .replace(/\.reshape\(/g, '.reshape(')
            .replace(/\.permute\(/g, '.transpose(')
            .replace(/\.contiguous\(\)/g, '')  // Not needed in JAX
            
            // Tensor creation
            .replace(/torch.zeros\(/g, 'jnp.zeros(')
            .replace(/torch.ones\(/g, 'jnp.ones(')
            .replace(/torch.randn\(/g, 'jnp.random.normal(key, ')
            .replace(/torch.tensor\(/g, 'jnp.array(')
            
            // Operations
            .replace(/\.mm\(/g, '.matmul(')
            .replace(/\.bmm\(/g, '.matmul(')
            .replace(/\.detach\(\)/g, '.copy()')  // No real equivalent, just use copy
            .replace(/\.cuda\(\)/g, '')  // Not needed in JAX
            .replace(/\.item\(\)/g, '.item()');

        return jaxCode;
    } catch (error) {
        console.error('Error converting PyTorch code to JAX:', error);
        return torchCode;  // Return original if conversion fails
    }
}

// Unified function to generate JAX/Flax code from PyTorch operations
export function to_jax(opType: string, params: Record<string, any> = {}, inputCode?: string): string {
    // Handle elementwise operations
    if (['add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'and', 'or', 'xor', 'not', 
        'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'identity'].includes(opType.toLowerCase())) {
        return getJaxElementwiseOpCode(opType, inputCode);
    }
    
    // Handle neural network modules
    if (nn_module_metadata[opType]) {
        return getTorchToFlaxCode(opType, params);
    }
    
    // Handle full code conversion
    if (inputCode) {
        return torchToJaxCode(inputCode);
    }
    
    throw new Error(`Unsupported operation type for JAX/Flax conversion: ${opType}`);
} 