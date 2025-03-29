interface ModuleMetadata {
    required_params: string[];
    optional_params: string[];
    code_generator: (params: Record<string, any>) => string;
}

export const nn_module_metadata: Record<string, ModuleMetadata> = {
    // Linear layers
    "Linear": {
        required_params: ["input_features", "output_features"],
        optional_params: ["bias"],
        code_generator: (params) => `nn.Linear(${params['input_features']}, ${params['output_features']}, bias=${params['bias'] ?? true})`
    },
    
    // Convolutional layers
    "Conv1D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.Conv1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },
    
    "Conv2D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.Conv2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },
    
    "Conv3D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.Conv3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },

    // Transposed convolutions
    "ConvTranspose1D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.ConvTranspose1d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },

    "ConvTranspose2D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.ConvTranspose2d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },

    "ConvTranspose3D": {
        required_params: ["in_channels", "out_channels", "kernel_size"],
        optional_params: ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        code_generator: (params) => `nn.ConvTranspose3d(${params['in_channels']}, ${params['out_channels']}, ${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? 1}, padding=${params['padding'] ?? 0}, ` +
            `output_padding=${params['output_padding'] ?? 0}, groups=${params['groups'] ?? 1}, ` +
            `bias=${params['bias'] ?? true}, dilation=${params['dilation'] ?? 1}, ` +
            `padding_mode='${params['padding_mode'] ?? 'zeros'}')`
    },

    // Pooling layers
    "MaxPool1D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool1d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`
    },

    "MaxPool2D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool2d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`
    },

    "MaxPool3D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        code_generator: (params) => `nn.MaxPool3d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `dilation=${params['dilation'] ?? 1}, return_indices=${params['return_indices'] ?? false}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false})`
    },

    "AvgPool1D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad"],
        code_generator: (params) => `nn.AvgPool1d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true})`
    },

    "AvgPool2D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        code_generator: (params) => `nn.AvgPool2d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true}, ` +
            `divisor_override=${params['divisor_override'] ?? 'None'})`
    },

    "AvgPool3D": {
        required_params: ["kernel_size"],
        optional_params: ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        code_generator: (params) => `nn.AvgPool3d(kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, padding=${params['padding'] ?? 0}, ` +
            `ceil_mode=${params['ceil_mode'] ?? false}, count_include_pad=${params['count_include_pad'] ?? true}, ` +
            `divisor_override=${params['divisor_override'] ?? 'None'})`
    },

    "LPPool1D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool1d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`
    },

    "LPPool2D": {
        required_params: ["norm_type", "kernel_size"],
        optional_params: ["stride", "ceil_mode"],
        code_generator: (params) => `nn.LPPool2d(norm_type=${params['norm_type']}, kernel_size=${params['kernel_size']}, ` +
            `stride=${params['stride'] ?? params['kernel_size']}, ceil_mode=${params['ceil_mode'] ?? false})`
    },

    // Adaptive pooling
    "AdaptiveAvgPool1D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool1d(output_size=${params['output_size']})`
    },

    "AdaptiveAvgPool2D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool2d(output_size=${params['output_size']})`
    },

    "AdaptiveAvgPool3D": {
        required_params: ["output_size"],
        optional_params: [],
        code_generator: (params) => `nn.AdaptiveAvgPool3d(output_size=${params['output_size']})`
    },

    "AdaptiveMaxPool1D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool1d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`
    },

    "AdaptiveMaxPool2D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool2d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`
    },

    "AdaptiveMaxPool3D": {
        required_params: ["output_size"],
        optional_params: ["return_indices"],
        code_generator: (params) => `nn.AdaptiveMaxPool3d(output_size=${params['output_size']}, return_indices=${params['return_indices'] ?? false})`
    },

    // Normalization layers
    "BatchNorm1D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.BatchNorm1d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true})`
    },

    "BatchNorm2D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.BatchNorm2d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true})`
    },

    "BatchNorm3D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.BatchNorm3d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? true}, track_running_stats=${params['track_running_stats'] ?? true})`
    },

    "LayerNorm": {
        required_params: ["normalized_shape"],
        optional_params: ["eps", "elementwise_affine"],
        code_generator: (params) => `nn.LayerNorm(normalized_shape=${params['normalized_shape']}, ` +
            `eps=${params['eps'] ?? 1e-5}, elementwise_affine=${params['elementwise_affine'] ?? true})`
    },

    "GroupNorm": {
        required_params: ["num_groups", "num_channels"],
        optional_params: ["eps", "affine"],
        code_generator: (params) => `nn.GroupNorm(num_groups=${params['num_groups']}, num_channels=${params['num_channels']}, ` +
            `eps=${params['eps'] ?? 1e-5}, affine=${params['affine'] ?? true})`
    },

    "InstanceNorm1D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.InstanceNorm1d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false})`
    },

    "InstanceNorm2D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.InstanceNorm2d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false})`
    },

    "InstanceNorm3D": {
        required_params: ["num_features"],
        optional_params: ["eps", "momentum", "affine", "track_running_stats"],
        code_generator: (params) => `nn.InstanceNorm3d(num_features=${params['num_features']}, ` +
            `eps=${params['eps'] ?? 1e-5}, momentum=${params['momentum'] ?? 0.1}, ` +
            `affine=${params['affine'] ?? false}, track_running_stats=${params['track_running_stats'] ?? false})`
    },

    // Dropout layers
    "Dropout": {
        required_params: [],
        optional_params: ["p", "inplace"],
        code_generator: (params) => `nn.Dropout(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`
    },

    "Dropout2D": {
        required_params: [],
        optional_params: ["p", "inplace"],
        code_generator: (params) => `nn.Dropout2d(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`
    },

    "Dropout3D": {
        required_params: [],
        optional_params: ["p", "inplace"],
        code_generator: (params) => `nn.Dropout3d(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`
    },

    "AlphaDropout": {
        required_params: [],
        optional_params: ["p", "inplace"],
        code_generator: (params) => `nn.AlphaDropout(p=${params['p'] ?? 0.5}, inplace=${params['inplace'] ?? false})`
    },

    // Activation functions
    "ReLU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.ReLU(inplace=${params['inplace'] ?? false})`
    },

    "LeakyReLU": {
        required_params: [],
        optional_params: ["negative_slope", "inplace"],
        code_generator: (params) => `nn.LeakyReLU(negative_slope=${params['negative_slope'] ?? 0.01}, inplace=${params['inplace'] ?? false})`
    },

    "Sigmoid": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Sigmoid()`
    },

    "Tanh": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Tanh()`
    },

    "ELU": {
        required_params: [],
        optional_params: ["alpha", "inplace"],
        code_generator: (params) => `nn.ELU(alpha=${params['alpha'] ?? 1.0}, inplace=${params['inplace'] ?? false})`
    },

    "SELU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.SELU(inplace=${params['inplace'] ?? false})`
    },

    "CELU": {
        required_params: [],
        optional_params: ["alpha", "inplace"],
        code_generator: (params) => `nn.CELU(alpha=${params['alpha'] ?? 1.0}, inplace=${params['inplace'] ?? false})`
    },

    "GELU": {
        required_params: [],
        optional_params: ["approximate"],
        code_generator: (params) => `nn.GELU(approximate='${params['approximate'] ?? 'none'}')`
    },

    "Softplus": {
        required_params: [],
        optional_params: ["beta", "threshold"],
        code_generator: (params) => `nn.Softplus(beta=${params['beta'] ?? 1}, threshold=${params['threshold'] ?? 20})`
    },

    "Softsign": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Softsign()`
    },

    "Softmax": {
        required_params: [],
        optional_params: ["dim"],
        code_generator: (params) => `nn.Softmax(dim=${params['dim'] ?? -1})`
    },

    "LogSoftmax": {
        required_params: [],
        optional_params: ["dim"],
        code_generator: (params) => `nn.LogSoftmax(dim=${params['dim'] ?? -1})`
    },

    "PReLU": {
        required_params: [],
        optional_params: ["num_parameters", "init"],
        code_generator: (params) => `nn.PReLU(num_parameters=${params['num_parameters'] ?? 1}, init=${params['init'] ?? 0.25})`
    },

    "Hardtanh": {
        required_params: [],
        optional_params: ["min_val", "max_val", "inplace"],
        code_generator: (params) => `nn.Hardtanh(min_val=${params['min_val'] ?? -1.0}, max_val=${params['max_val'] ?? 1.0}, inplace=${params['inplace'] ?? false})`
    },

    "Hardshrink": {
        required_params: [],
        optional_params: ["lambd"],
        code_generator: (params) => `nn.Hardshrink(lambd=${params['lambd'] ?? 0.5})`
    },

    "Hardsigmoid": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Hardsigmoid(inplace=${params['inplace'] ?? false})`
    },

    "Hardswish": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Hardswish(inplace=${params['inplace'] ?? false})`
    },

    "RReLU": {
        required_params: [],
        optional_params: ["lower", "upper", "inplace"],
        code_generator: (params) => `nn.RReLU(lower=${params['lower'] ?? 1/8}, upper=${params['upper'] ?? 1/3}, inplace=${params['inplace'] ?? false})`
    },

    "Softshrink": {
        required_params: [],
        optional_params: ["lambd"],
        code_generator: (params) => `nn.Softshrink(lambd=${params['lambd'] ?? 0.5})`
    },

    "Tanhshrink": {
        required_params: [],
        optional_params: [],
        code_generator: () => `nn.Tanhshrink()`
    },

    "Threshold": {
        required_params: ["threshold", "value"],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Threshold(threshold=${params['threshold']}, value=${params['value']}, inplace=${params['inplace'] ?? false})`
    },

    "ReLU6": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.ReLU6(inplace=${params['inplace'] ?? false})`
    },

    "SiLU": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.SiLU(inplace=${params['inplace'] ?? false})`
    },

    "Mish": {
        required_params: [],
        optional_params: ["inplace"],
        code_generator: (params) => `nn.Mish(inplace=${params['inplace'] ?? false})`
    },

    // Reshape operations
    "Reshape": {
        required_params: ["shape"],
        optional_params: [],
        code_generator: (params) => `torch.reshape(${params['shape'].join(', ')})`
    },

    "Permute": {
        required_params: ["dims"],
        optional_params: [],
        code_generator: (params) => `torch.permute(${params['dims'].join(', ')})`
    },

    "Flatten": {
        required_params: [],
        optional_params: ["start_dim", "end_dim"],
        code_generator: (params) => `nn.Flatten(start_dim=${params['start_dim'] ?? 1}, end_dim=${params['end_dim'] ?? -1})`
    },

    "Unflatten": {
        required_params: ["dim", "unflattened_size"],
        optional_params: [],
        code_generator: (params) => `nn.Unflatten(dim=${params['dim']}, unflattened_size=${params['unflattened_size']})`
    }
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

    if (module_type in nn_module_metadata) {
        return nn_module_metadata[module_type].code_generator(params);
    } else {
        throw new Error(`Unknown module type: ${module_type}`);
    }
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
        default:
            throw new Error(`Unknown elementwise operation type: ${opType}`);
    }
} 