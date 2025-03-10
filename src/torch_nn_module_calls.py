"""
PyTorch neural network module metadata and code generation utilities.
Contains information about required and optional parameters for each module type,
and functions to generate properly formatted PyTorch code.
"""

# Dictionary defining required parameters and code generation for PyTorch modules
nn_module_metadata = {
    # Linear layers
    "Linear": {
        "required_params": ["input_features", "output_features"],
        "optional_params": ["bias"],
        "code_generator": lambda params: f"nn.Linear({params['input_features']}, {params['output_features']}, bias={params.get('bias', True)})"
    },
    
    # Convolutional layers
    "Conv1D": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.Conv1d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    "Conv2D": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.Conv2d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    "Conv3D": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.Conv3d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    # Transposed convolutions
    "Conv1DTranspose": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.ConvTranspose1d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, output_padding={params.get('output_padding', 0)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, dilation={params.get('dilation', 1)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    "Conv2DTranspose": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.ConvTranspose2d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, output_padding={params.get('output_padding', 0)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, dilation={params.get('dilation', 1)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    "Conv3DTranspose": {
        "required_params": ["in_channels", "out_channels", "kernel_size"],
        "optional_params": ["stride", "padding", "output_padding", "dilation", "groups", "bias", "padding_mode"],
        "code_generator": lambda params: f"nn.ConvTranspose3d({params['in_channels']}, {params['out_channels']}, {params['kernel_size']}, stride={params.get('stride', 1)}, padding={params.get('padding', 0)}, output_padding={params.get('output_padding', 0)}, groups={params.get('groups', 1)}, bias={params.get('bias', True)}, dilation={params.get('dilation', 1)}, padding_mode='{params.get('padding_mode', 'zeros')}')"
    },
    
    # Pooling layers
    "MaxPool1D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        "code_generator": lambda params: f"nn.MaxPool1d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, return_indices={params.get('return_indices', False)}, ceil_mode={params.get('ceil_mode', False)})"
    },
    
    "MaxPool2D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        "code_generator": lambda params: f"nn.MaxPool2d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, return_indices={params.get('return_indices', False)}, ceil_mode={params.get('ceil_mode', False)})"
    },
    
    "MaxPool3D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "dilation", "return_indices", "ceil_mode"],
        "code_generator": lambda params: f"nn.MaxPool3d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, return_indices={params.get('return_indices', False)}, ceil_mode={params.get('ceil_mode', False)})"
    },
    
    "AvgPool1D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "ceil_mode", "count_include_pad"],
        "code_generator": lambda params: f"nn.AvgPool1d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, ceil_mode={params.get('ceil_mode', False)}, count_include_pad={params.get('count_include_pad', True)})"
    },
    
    "AvgPool2D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        "code_generator": lambda params: f"nn.AvgPool2d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, ceil_mode={params.get('ceil_mode', False)}, count_include_pad={params.get('count_include_pad', True)}, divisor_override={params.get('divisor_override', 'None')})"
    },
    
    "AvgPool3D": {
        "required_params": ["kernel_size"],
        "optional_params": ["stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
        "code_generator": lambda params: f"nn.AvgPool3d(kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, padding={params.get('padding', 0)}, ceil_mode={params.get('ceil_mode', False)}, count_include_pad={params.get('count_include_pad', True)}, divisor_override={params.get('divisor_override', 'None')})"
    },
    
    "LPPool1D": {
        "required_params": ["norm_type", "kernel_size"],
        "optional_params": ["stride", "ceil_mode"],
        "code_generator": lambda params: f"nn.LPPool1d(norm_type={params['norm_type']}, kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, ceil_mode={params.get('ceil_mode', False)})"
    },
    
    "LPPool2D": {
        "required_params": ["norm_type", "kernel_size"],
        "optional_params": ["stride", "ceil_mode"],
        "code_generator": lambda params: f"nn.LPPool2d(norm_type={params['norm_type']}, kernel_size={params['kernel_size']}, stride={params.get('stride', params['kernel_size'])}, ceil_mode={params.get('ceil_mode', False)})"
    },
    
    # Adaptive pooling
    "AdaptiveAvgPool1D": {
        "required_params": ["output_size"],
        "optional_params": [],
        "code_generator": lambda params: f"nn.AdaptiveAvgPool1d(output_size={params['output_size']})"
    },
    "AdaptiveAvgPool2D": {
        "required_params": ["output_size"],
        "optional_params": [],
        "code_generator": lambda params: f"nn.AdaptiveAvgPool2d(output_size={params['output_size']})"
    },
    "AdaptiveAvgPool3D": {
        "required_params": ["output_size"],
        "optional_params": [],
        "code_generator": lambda params: f"nn.AdaptiveAvgPool3d(output_size={params['output_size']})"
    },
    
    "AdaptiveMaxPool1D": {
        "required_params": ["output_size"],
        "optional_params": ["return_indices"],
        "code_generator": lambda params: f"nn.AdaptiveMaxPool1d(output_size={params['output_size']}, return_indices={params.get('return_indices', False)})"
    },
    "AdaptiveMaxPool2D": {
        "required_params": ["output_size"],
        "optional_params": ["return_indices"],
        "code_generator": lambda params: f"nn.AdaptiveMaxPool2d(output_size={params['output_size']}, return_indices={params.get('return_indices', False)})"
    },
    "AdaptiveMaxPool3D": {
        "required_params": ["output_size"],
        "optional_params": ["return_indices"],
        "code_generator": lambda params: f"nn.AdaptiveMaxPool3d(output_size={params['output_size']}, return_indices={params.get('return_indices', False)})"
    },
    
    # Normalization layers
    "BatchNorm1D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.BatchNorm1d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', True)}, track_running_stats={params.get('track_running_stats', True)})"
    },
    
    "BatchNorm2D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.BatchNorm2d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', True)}, track_running_stats={params.get('track_running_stats', True)})"
    },
    
    "BatchNorm3D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.BatchNorm3d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', True)}, track_running_stats={params.get('track_running_stats', True)})"
    },
    
    "LayerNorm": {
        "required_params": ["normalized_shape"],
        "optional_params": ["eps", "elementwise_affine"],
        "code_generator": lambda params: f"nn.LayerNorm(normalized_shape={params['normalized_shape']}, eps={params.get('eps', 1e-5)}, elementwise_affine={params.get('elementwise_affine', True)})"
    },
    
    "GroupNorm": {
        "required_params": ["num_groups", "num_channels"],
        "optional_params": ["eps", "affine"],
        "code_generator": lambda params: f"nn.GroupNorm(num_groups={params['num_groups']}, num_channels={params['num_channels']}, eps={params.get('eps', 1e-5)}, affine={params.get('affine', True)})"
    },
    
    "InstanceNorm1D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.InstanceNorm1d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', False)}, track_running_stats={params.get('track_running_stats', False)})"
    },
    
    "InstanceNorm2D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.InstanceNorm2d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', False)}, track_running_stats={params.get('track_running_stats', False)})"
    },
    
    "InstanceNorm3D": {
        "required_params": ["num_features"],
        "optional_params": ["eps", "momentum", "affine", "track_running_stats"],
        "code_generator": lambda params: f"nn.InstanceNorm3d(num_features={params['num_features']}, eps={params.get('eps', 1e-5)}, momentum={params.get('momentum', 0.1)}, affine={params.get('affine', False)}, track_running_stats={params.get('track_running_stats', False)})"
    },
    
    # Dropout
    "Dropout": {
        "required_params": [],
        "optional_params": ["p", "inplace"],
        "code_generator": lambda params: f"nn.Dropout(p={params.get('p', 0.5)}, inplace={params.get('inplace', False)})"
    },
    "Dropout2D": {
        "required_params": [],
        "optional_params": ["p", "inplace"],
        "code_generator": lambda params: f"nn.Dropout2d(p={params.get('p', 0.5)}, inplace={params.get('inplace', False)})"
    },
    "Dropout3D": {
        "required_params": [],
        "optional_params": ["p", "inplace"],
        "code_generator": lambda params: f"nn.Dropout3d(p={params.get('p', 0.5)}, inplace={params.get('inplace', False)})"
    },
    "AlphaDropout": {
        "required_params": [],
        "optional_params": ["p", "inplace"],
        "code_generator": lambda params: f"nn.AlphaDropout(p={params.get('p', 0.5)}, inplace={params.get('inplace', False)})"
    },
    
    # Activation functions
    "ReLU": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.ReLU(inplace={params.get('inplace', False)})"
    },
    "LeakyReLU": {
        "required_params": [],
        "optional_params": ["negative_slope", "inplace"],
        "code_generator": lambda params: f"nn.LeakyReLU(negative_slope={params.get('negative_slope', 0.01)}, inplace={params.get('inplace', False)})"
    },
    "Sigmoid": {
        "required_params": [],
        "optional_params": [],
        "code_generator": lambda params: f"nn.Sigmoid()"
    },
    "Tanh": {
        "required_params": [],
        "optional_params": [],
        "code_generator": lambda params: f"nn.Tanh()"
    },
    "ELU": {
        "required_params": [],
        "optional_params": ["alpha", "inplace"],
        "code_generator": lambda params: f"nn.ELU(alpha={params.get('alpha', 1.0)}, inplace={params.get('inplace', False)})"
    },
    "SELU": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.SELU(inplace={params.get('inplace', False)})"
    },
    "CELU": {
        "required_params": [],
        "optional_params": ["alpha", "inplace"],
        "code_generator": lambda params: f"nn.CELU(alpha={params.get('alpha', 1.0)}, inplace={params.get('inplace', False)})"
    },
    "GELU": {
        "required_params": [],
        "optional_params": ["approximate"],
        "code_generator": lambda params: f"nn.GELU(approximate={params.get('approximate', 'none')})"
    },
    "Softplus": {
        "required_params": [],
        "optional_params": ["beta", "threshold"],
        "code_generator": lambda params: f"nn.Softplus(beta={params.get('beta', 1)}, threshold={params.get('threshold', 20)})"
    },
    "Softsign": {
        "required_params": [],
        "optional_params": [],
        "code_generator": lambda params: f"nn.Softsign()"
    },
    "Softmax": {
        "required_params": [],
        "optional_params": ["dim"],
        "code_generator": lambda params: f"nn.Softmax(dim={params.get('dim', -1)})"
    },
    "LogSoftmax": {
        "required_params": [],
        "optional_params": ["dim"],
        "code_generator": lambda params: f"nn.LogSoftmax(dim={params.get('dim', -1)})"
    },
    "PReLU": {
        "required_params": [],
        "optional_params": ["num_parameters", "init"],
        "code_generator": lambda params: f"nn.PReLU(num_parameters={params.get('num_parameters', 1)}, init={params.get('init', 0.25)})"
    },
    "Hardtanh": {
        "required_params": [],
        "optional_params": ["min_val", "max_val", "inplace"],
        "code_generator": lambda params: f"nn.Hardtanh(min_val={params.get('min_val', -1.0)}, max_val={params.get('max_val', 1.0)}, inplace={params.get('inplace', False)})"
    },
    "Hardshrink": {
        "required_params": [],
        "optional_params": ["lambd"],
        "code_generator": lambda params: f"nn.Hardshrink(lambd={params.get('lambd', 0.5)})"
    },
    "Hardsigmoid": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.Hardsigmoid(inplace={params.get('inplace', False)})"
    },
    "Hardswish": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.Hardswish(inplace={params.get('inplace', False)})"
    },
    "RReLU": {
        "required_params": [],
        "optional_params": ["lower", "upper", "inplace"],
        "code_generator": lambda params: f"nn.RReLU(lower={params.get('lower', 1/8)}, upper={params.get('upper', 1/3)}, inplace={params.get('inplace', False)})"
    },
    "Softshrink": {
        "required_params": [],
        "optional_params": ["lambd"],
        "code_generator": lambda params: f"nn.Softshrink(lambd={params.get('lambd', 0.5)})"
    },
    "Tanhshrink": {
        "required_params": [],
        "optional_params": [],
        "code_generator": lambda params: f"nn.Tanhshrink()"
    },
    "Threshold": {
        "required_params": ["threshold", "value"],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.Threshold(threshold={params['threshold']}, value={params['value']}, inplace={params.get('inplace', False)})"
    },
    "ReLU6": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.ReLU6(inplace={params.get('inplace', False)})"
    },
    "SiLU": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.SiLU(inplace={params.get('inplace', False)})"
    },
    "Mish": {
        "required_params": [],
        "optional_params": ["inplace"],
        "code_generator": lambda params: f"nn.Mish(inplace={params.get('inplace', False)})"
    },
    
    # Reshape operations
    "Flatten": {
        "required_params": [],
        "optional_params": ["start_dim", "end_dim"],
        "code_generator": lambda params: f"nn.Flatten(start_dim={params.get('start_dim', 1)}, end_dim={params.get('end_dim', -1)})"
    },
    "Unflatten": {
        "required_params": ["dim", "unflattened_size"],
        "optional_params": [],
        "code_generator": lambda params: f"nn.Unflatten(dim={params['dim']}, unflattened_size={params['unflattened_size']})"
    },
    
    # Recurrent layers
    "RNN": {
        "required_params": ["input_size", "hidden_size"],
        "optional_params": ["num_layers", "nonlinearity", "bias", "batch_first", "dropout", "bidirectional"],
        "code_generator": lambda params: f"nn.RNN(input_size={params['input_size']}, hidden_size={params['hidden_size']}, num_layers={params.get('num_layers', 1)}, nonlinearity='{params.get('nonlinearity', 'tanh')}', bias={params.get('bias', True)}, batch_first={params.get('batch_first', False)}, dropout={params.get('dropout', 0)}, bidirectional={params.get('bidirectional', False)})"
    },
    
    "LSTM": {
        "required_params": ["input_size", "hidden_size"],
        "optional_params": ["num_layers", "bias", "batch_first", "dropout", "bidirectional"],
        "code_generator": lambda params: f"nn.LSTM(input_size={params['input_size']}, hidden_size={params['hidden_size']}, num_layers={params.get('num_layers', 1)}, bias={params.get('bias', True)}, batch_first={params.get('batch_first', False)}, dropout={params.get('dropout', 0)}, bidirectional={params.get('bidirectional', False)})"
    },
    
    "GRU": {
        "required_params": ["input_size", "hidden_size"],
        "optional_params": ["num_layers", "bias", "batch_first", "dropout", "bidirectional"],
        "code_generator": lambda params: f"nn.GRU(input_size={params['input_size']}, hidden_size={params['hidden_size']}, num_layers={params.get('num_layers', 1)}, bias={params.get('bias', True)}, batch_first={params.get('batch_first', False)}, dropout={params.get('dropout', 0)}, bidirectional={params.get('bidirectional', False)})"
    },
    
    # Transformers
    "TransformerEncoderLayer": {
        "required_params": ["d_model", "nhead"],
        "optional_params": ["dim_feedforward", "dropout", "activation", "batch_first"],
        "code_generator": lambda params: f"nn.TransformerEncoderLayer(d_model={params['d_model']}, nhead={params['nhead']}, dim_feedforward={params.get('dim_feedforward', 2048)}, dropout={params.get('dropout', 0.1)}, activation={params.get('activation', 'relu')}, batch_first={params.get('batch_first', False)})"
    },
    
    # Embeddings
    "Embedding": {
        "required_params": ["num_embeddings", "embedding_dim"],
        "optional_params": ["padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"],
        "code_generator": lambda params: f"nn.Embedding(num_embeddings={params['num_embeddings']}, embedding_dim={params['embedding_dim']}, padding_idx={params.get('padding_idx', 'None')}, max_norm={params.get('max_norm', 'None')}, norm_type={params.get('norm_type', 2.0)}, scale_grad_by_freq={params.get('scale_grad_by_freq', False)}, sparse={params.get('sparse', False)})"
    }
}

# For backward compatibility - simplified dictionary that maps module types directly to code generators
nn_module_dict = {key: value["code_generator"] for key, value in nn_module_metadata.items()}


# Utility functions
def validate_params(module_type, params):
    """
    Validate that all required parameters for a module type are present.
    
    Args:
        module_type (str): The PyTorch module type
        params (dict): The parameters dictionary
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails with missing parameters
    """
    if module_type not in nn_module_metadata:
        raise ValueError(f"Unknown module type: {module_type}")
    
    metadata = nn_module_metadata[module_type]
    missing_params = []
    
    for param in metadata["required_params"]:
        if param not in params:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters for {module_type}: {', '.join(missing_params)}")
    
    return True


def get_torch_code(module_type, params):
    """
    Generate PyTorch code for a module with validation.
    
    Args:
        module_type (str): The PyTorch module type
        params (dict): The parameters dictionary
        
    Returns:
        str: The PyTorch code for this module
        
    Raises:
        ValueError: If validation fails or unknown module type
    """
    validate_params(module_type, params)
    
    if module_type in nn_module_metadata:
        return nn_module_metadata[module_type]["code_generator"](params)
    else:
        raise ValueError(f"Unknown module type: {module_type}") 