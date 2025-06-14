import { ModuleMetadata, ParamFieldMetadata } from './types';

// Common parameter templates
const commonParams: Record<string, ParamFieldMetadata> = {
    inplace: {
        label: 'Inplace',
        description: 'If set to true, will do the operation in-place',
        type: 'option',
        options: ['true', 'false'],
        default: 'false'
    },
    alpha: {
        label: 'Alpha',
        description: 'The alpha value for the activation function',
        type: 'number',
        default: 1.0
    },
    dim: {
        label: 'Dimension',
        description: 'Dimension along which to apply the function',
        type: 'number',
        default: -1
    }
};

// Factory for simple activation functions with just inplace option
function createSimpleActivation(name: string, description: string): ModuleMetadata {
    return {
        label: name,
        description,
        category: 'Activation',
        paramFields: {
            inplace: commonParams.inplace
        }
    };
}

// Factory for parameterless activation functions
function createNoParamActivation(name: string, description: string): ModuleMetadata {
    return {
        label: name,
        description,
        category: 'Activation',
        paramFields: {}
    };
}

// Generate all activation module definitions
export const activationModules: Record<string, ModuleMetadata> = {
    // Simple ReLU family activations with inplace option
    'ReLU': createSimpleActivation('ReLU', 'Applies the rectified linear unit function element-wise'),
    'ReLU6': createSimpleActivation('ReLU6', 'Applies the ReLU6 function element-wise (min(max(0,x), 6))'),
    'SELU': createSimpleActivation('SELU', 'Applies the scaled exponential linear unit function'),
    'Hardswish': createSimpleActivation('Hardswish', 'Applies the hardswish function element-wise'),
    'Hardsigmoid': createSimpleActivation('Hardsigmoid', 'Applies the hardsigmoid function element-wise'),
    'SiLU': createSimpleActivation('SiLU', 'Applies the Sigmoid Linear Unit (SiLU/Swish) function'),
    'Mish': createSimpleActivation('Mish', 'Applies the Mish function, a self-regularized non-monotonic activation'),
  
    // No parameter activations
    'Sigmoid': createNoParamActivation('Sigmoid', 'Applies the sigmoid function element-wise'),
    'Tanh': createNoParamActivation('Tanh', 'Applies the hyperbolic tangent function element-wise'),
    'Softsign': createNoParamActivation('Softsign', 'Applies the softsign function element-wise'),
    'Tanhshrink': createNoParamActivation('Tanhshrink', 'Applies the tanhshrink function element-wise'),
  
    // Custom activations with specific parameters
    'LeakyReLU': {
        label: 'Leaky ReLU',
        description: 'Applies leaky rectified linear unit function with small slope for negative inputs',
        category: 'Activation',
        paramFields: {
            negative_slope: {
                label: 'Negative Slope',
                description: 'Controls the angle of the negative slope',
                type: 'number',
                default: 0.01
            },
            inplace: commonParams.inplace
        }
    },
  
    'ELU': {
        label: 'ELU',
        description: 'Applies the exponential linear unit function element-wise',
        category: 'Activation',
        paramFields: {
            alpha: {
                ...commonParams.alpha,
                description: 'The alpha value for the ELU formulation'
            },
            inplace: commonParams.inplace
        }
    },
  
    'CELU': {
        label: 'CELU',
        description: 'Applies the continuously differentiable exponential linear unit',
        category: 'Activation',
        paramFields: {
            alpha: {
                ...commonParams.alpha,
                description: 'The α value for the CELU formulation'
            },
            inplace: commonParams.inplace
        }
    },
  
    'GELU': {
        label: 'GELU',
        description: 'Applies the Gaussian error linear unit function',
        category: 'Activation',
        paramFields: {
            approximate: {
                label: 'Approximate',
                description: 'The approximation method to use',
                type: 'option',
                options: ['none', 'tanh'],
                default: 'none'
            }
        }
    },
  
    'Softplus': {
        label: 'Softplus',
        description: 'Applies the softplus function element-wise',
        category: 'Activation',
        paramFields: {
            beta: {
                label: 'Beta',
                description: 'The beta value for the softplus formulation',
                type: 'number',
                default: 1
            },
            threshold: {
                label: 'Threshold',
                description: 'Values above this revert to a linear function',
                type: 'number',
                default: 20
            }
        }
    },

    'Softmax': {
        label: 'Softmax',
        description: 'Applies the softmax function to normalize inputs into a probability distribution',
        category: 'Activation',
        paramFields: {
            dim: {
                ...commonParams.dim,
                description: 'Dimension along which softmax will be computed'
            }
        }
    },

    'LogSoftmax': {
        label: 'Log Softmax',
        description: 'Applies the log softmax function to normalize inputs',
        category: 'Activation',
        paramFields: {
            dim: {
                ...commonParams.dim,
                description: 'Dimension along which log_softmax will be computed'
            }
        }
    },

    'PReLU': {
        label: 'PReLU',
        description: 'Applies the parametric rectified linear unit function',
        category: 'Activation',
        paramFields: {
            num_parameters: {
                label: 'Number of Parameters',
                description: 'Number of a to learn, can be 1 or input size',
                type: 'number',
                default: 1
            },
            init: {
                label: 'Init Value',
                description: 'Initial value of a',
                type: 'number',
                default: 0.25
            }
        }
    },
    'Hardtanh': {
        label: 'Hardtanh',
        description: 'Applies the HardTanh function element-wise',
        category: 'Activation',
        paramFields: {
            min_val: {
                label: 'Minimum Value',
                description: 'Minimum value of the linear region range',
                type: 'number',
                default: -1
            },
            max_val: {
                label: 'Maximum Value',
                description: 'Maximum value of the linear region range',
                type: 'number',
                default: 1
            },
        }
    },
    'Hardshrink': {
        label: 'Hardshrink',
        description: 'Applies the hard shrinkage function element-wise',
        category: 'Activation',
        paramFields: {
            lambda: {
                label: 'Lambda',
                description: 'The lambda value for the Hardshrink formulation',
                type: 'number',
                default: 0.5
            }
        }
    },
    'RReLU': {
        label: 'RReLU',
        description: 'Applies the randomized rectified linear unit function element-wise',
        category: 'Activation',
        paramFields: {
            lower: {
                label: 'Lower Bound',
                description: 'Lower bound of the uniform distribution',
                type: 'number',
                default: 0.125
            },
            upper: {
                label: 'Upper Bound',
                description: 'Upper bound of the uniform distribution',
                type: 'number',
                default: 0.3333333333333333
            }
        }
    },
    'Softshrink': {
        label: 'Softshrink',
        description: 'Applies the soft shrinkage function element-wise',
        category: 'Activation',
        paramFields: {
            lambda: {
                label: 'Lambda',
                description: 'The lambda value for the Softshrink formulation',
                type: 'number',
                default: 0.5
            }
        }
    }
};
