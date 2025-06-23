import { ModuleDef } from './types';

export const ReLU: ModuleDef = {
    label: 'ReLU',
    description: 'Applies the rectified linear unit function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.ReLU()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const LeakyReLU: ModuleDef = {
    label: 'Leaky ReLU',
    description: 'Applies leaky rectified linear unit function with small slope for negative inputs',
    category: 'Activation',
    moduleType: "Op",
    params: {
        negative_slope: {
            label: 'Negative Slope',
            description: 'Controls the angle of the negative slope',
            type: 'number',
            default: 0.01,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.LeakyReLU(negative_slope=${params.negative_slope ?? 0.01})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Sigmoid: ModuleDef = {
    label: 'Sigmoid',
    description: 'Applies the sigmoid function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Sigmoid()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Tanh: ModuleDef = {
    label: 'Tanh',
    description: 'Applies the hyperbolic tangent function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Tanh()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const ELU: ModuleDef = {
    label: 'ELU',
    description: 'Applies the exponential linear unit function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        alpha: {
            label: 'Alpha',
            description: 'The alpha value for the ELU formulation',
            type: 'number',
            default: 1.0,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.ELU(alpha=${params.alpha ?? 1.0})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const SELU: ModuleDef = {
    label: 'SELU',
    description: 'Applies the scaled exponential linear unit function',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.SELU()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const CELU: ModuleDef = {
    label: 'CELU',
    description: 'Applies the continuously differentiable exponential linear unit',
    category: 'Activation',
    moduleType: "Op",
    params: {
        alpha: {
            label: 'Alpha',
            description: 'The α value for the CELU formulation',
            type: 'number',
            default: 1.0,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.CELU(alpha=${params.alpha ?? 1.0})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const GELU: ModuleDef = {
    label: 'GELU',
    description: 'Applies the Gaussian error linear unit function',
    category: 'Activation',
    moduleType: "Op",
    params: {
        approximate: {
            label: 'Approximate',
            description: 'The approximation method to use',
            type: 'option',
            options: ['none', 'tanh'],
            default: 'none',
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.GELU(approximate='${params.approximate ?? 'none'}')`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Softplus: ModuleDef = {
    label: 'Softplus',
    description: 'Applies the softplus function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        beta: {
            label: 'Beta',
            description: 'The beta value for the softplus formulation',
            type: 'number',
            default: 1,
            required: false
        },
        threshold: {
            label: 'Threshold',
            description: 'Values above this revert to a linear function',
            type: 'number',
            default: 20,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Softplus(beta=${params.beta ?? 1}, threshold=${params.threshold ?? 20})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Softsign: ModuleDef = {
    label: 'Softsign',
    description: 'Applies the softsign function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Softsign()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Softmax: ModuleDef = {
    label: 'Softmax',
    description: 'Applies the softmax function to normalize inputs into a probability distribution',
    category: 'Activation',
    moduleType: "Op",
    params: {
        dim: {
            label: 'Dimension',
            description: 'Dimension along which softmax will be computed',
            type: 'number',
            default: -1,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Softmax(dim=${params.dim ?? -1})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        if (inShape.length === 0) {
            errors.push('Softmax requires at least 1D input tensor');
            return errors;
        }
        
        const dim = params.dim ?? -1;
        const normalizedDim = dim < 0 ? inShape.length + dim : dim;
        
        if (normalizedDim < 0 || normalizedDim >= inShape.length) {
            errors.push(`Softmax dimension ${dim} is out of bounds for input shape [${inShape.join(', ')}]`);
        }
        
        return errors;
    }
};

export const LogSoftmax: ModuleDef = {
    label: 'Log Softmax',
    description: 'Applies the log softmax function to normalize inputs',
    category: 'Activation',
    moduleType: "Op",
    params: {
        dim: {
            label: 'Dimension',
            description: 'Dimension along which log_softmax will be computed',
            type: 'number',
            default: -1,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.LogSoftmax(dim=${params.dim ?? -1})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        if (inShape.length === 0) {
            errors.push('LogSoftmax requires at least 1D input tensor');
            return errors;
        }
        
        const dim = params.dim ?? -1;
        const normalizedDim = dim < 0 ? inShape.length + dim : dim;
        
        if (normalizedDim < 0 || normalizedDim >= inShape.length) {
            errors.push(`LogSoftmax dimension ${dim} is out of bounds for input shape [${inShape.join(', ')}]`);
        }
        
        return errors;
    }
};

export const PReLU: ModuleDef = {
    label: 'PReLU',
    description: 'Applies the parametric rectified linear unit function',
    category: 'Activation',
    moduleType: "Op",
    params: {
        num_parameters: {
            label: 'Number of Parameters',
            description: 'Number of a to learn, can be 1 or input size',
            type: 'number',
            default: 1,
            required: false
        },
        init: {
            label: 'Init Value',
            description: 'Initial value of a',
            type: 'number',
            default: 0.25,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.PReLU(num_parameters=${params.num_parameters ?? 1}, init=${params.init ?? 0.25})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Hardtanh: ModuleDef = {
    label: 'Hardtanh',
    description: 'Applies the HardTanh function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        min_val: {
            label: 'Minimum Value',
            description: 'Minimum value of the linear region range',
            type: 'number',
            default: -1,
            required: false
        },
        max_val: {
            label: 'Maximum Value',
            description: 'Maximum value of the linear region range',
            type: 'number',
            default: 1,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Hardtanh(min_val=${params.min_val ?? -1.0}, max_val=${params.max_val ?? 1.0})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Hardshrink: ModuleDef = {
    label: 'Hardshrink',
    description: 'Applies the hard shrinkage function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        lambda: {
            label: 'Lambda',
            description: 'The lambda value for the Hardshrink formulation',
            type: 'number',
            default: 0.5,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Hardshrink(lambd=${params.lambda ?? 0.5})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Hardsigmoid: ModuleDef = {
    label: 'Hardsigmoid',
    description: 'Applies the hardsigmoid function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Hardsigmoid()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Hardswish: ModuleDef = {
    label: 'Hardswish',
    description: 'Applies the hardswish function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Hardswish()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const RReLU: ModuleDef = {
    label: 'RReLU',
    description: 'Applies the randomized rectified linear unit function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        lower: {
            label: 'Lower Bound',
            description: 'Lower bound of the uniform distribution',
            type: 'number',
            default: 0.125,
            required: false
        },
        upper: {
            label: 'Upper Bound',
            description: 'Upper bound of the uniform distribution',
            type: 'number',
            default: 0.3333333333333333,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.RReLU(lower=${params.lower ?? 1/8}, upper=${params.upper ?? 1/3})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Softshrink: ModuleDef = {
    label: 'Softshrink',
    description: 'Applies the soft shrinkage function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {
        lambda: {
            label: 'Lambda',
            description: 'The lambda value for the Softshrink formulation',
            type: 'number',
            default: 0.5,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Softshrink(lambd=${params.lambda ?? 0.5})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Tanhshrink: ModuleDef = {
    label: 'Tanhshrink',
    description: 'Applies the tanhshrink function element-wise',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Tanhshrink()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Threshold: ModuleDef = {
    label: 'Threshold',
    description: 'Thresholds each element of the input tensor',
    category: 'Activation',
    moduleType: "Op",
    params: {
        threshold: {
            label: 'Threshold',
            description: 'The value to threshold at',
            type: 'number',
            required: true
        },
        value: {
            label: 'Value',
            description: 'The value to replace with',
            type: 'number',
            required: true
        }
    },
    emitPytorchModule: (params) => `nn.Threshold(threshold=${params.threshold}, value=${params.value})`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const ReLU6: ModuleDef = {
    label: 'ReLU6',
    description: 'Applies the ReLU6 function element-wise (min(max(0,x), 6))',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.ReLU6()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const SiLU: ModuleDef = {
    label: 'SiLU',
    description: 'Applies the Sigmoid Linear Unit (SiLU/Swish) function',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.SiLU()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};

export const Mish: ModuleDef = {
    label: 'Mish',
    description: 'Applies the Mish function, a self-regularized non-monotonic activation',
    category: 'Activation',
    moduleType: "Op",
    params: {},
    emitPytorchModule: (_params) => `nn.Mish()`,
    inferOutputShape: (inShape) => inShape,
    validateInputShape: (_inShape) => []
};
