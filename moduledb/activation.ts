import { ModuleDef } from './types';

export const ReLU: ModuleDef = {
    label: 'ReLU',
    description: 'Applies the rectified linear unit function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {},
    toPytorchExpr: (_params) => `nn.ReLU()`,
    shapeInference: (inShape) => inShape // ReLU preserves shape
};

export const Sigmoid: ModuleDef = {
    label: 'Sigmoid',
    description: 'Applies the sigmoid function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {},
    toPytorchExpr: (_params) => `nn.Sigmoid()`,
    shapeInference: (inShape) => inShape // Sigmoid preserves shape
};

export const Tanh: ModuleDef = {
    label: 'Tanh',
    description: 'Applies the hyperbolic tangent function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {},
    toPytorchExpr: (_params) => `nn.Tanh()`,
    shapeInference: (inShape) => inShape // Tanh preserves shape
};

export const LeakyReLU: ModuleDef = {
    label: 'LeakyReLU',
    description: 'Applies the leaky rectified linear unit function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {
        negative_slope: {
            label: 'Negative Slope',
            description: 'Controls the angle of the negative slope',
            type: 'number',
            default: 0.01,
            required: false
        }
    },
    toPytorchExpr: (params) => `nn.LeakyReLU(negative_slope=${params.negative_slope || 0.01})`,
    shapeInference: (inShape) => inShape // LeakyReLU preserves shape
};

export const ELU: ModuleDef = {
    label: 'ELU',
    description: 'Applies the exponential linear unit function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {
        alpha: {
            label: 'Alpha',
            description: 'The alpha value for the ELU formulation',
            type: 'number',
            default: 1.0,
            required: false
        }
    },
    toPytorchExpr: (params) => `nn.ELU(alpha=${params.alpha || 1.0})`,
    shapeInference: (inShape) => inShape // ELU preserves shape
};

export const SELU: ModuleDef = {
    label: 'SELU',
    description: 'Applies the scaled exponential linear unit function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {},
    toPytorchExpr: (_params) => `nn.SELU()`,
    shapeInference: (inShape) => inShape // SELU preserves shape
};

export const Swish: ModuleDef = {
    label: 'Swish',
    description: 'Applies the Swish function element-wise (x * sigmoid(x))',
    category: 'Activation',
    moduleType: "op",
    params: {},
    toPytorchExpr: (_params) => `nn.SiLU()`, // SiLU is PyTorch's implementation of Swish
    shapeInference: (inShape) => inShape // Swish preserves shape
};

export const GELU: ModuleDef = {
    label: 'GELU',
    description: 'Applies the Gaussian Error Linear Unit function element-wise',
    category: 'Activation',
    moduleType: "op",
    params: {
        approximate: {
            label: 'Approximate',
            description: 'Whether to use the approximate form of GELU',
            type: 'string',
            default: 'none',
            required: false
        }
    },
    toPytorchExpr: (params) => {
        const approx = params.approximate || 'none';
        return approx === 'none' ? `nn.GELU()` : `nn.GELU(approximate='${approx}')`;
    },
    shapeInference: (inShape) => inShape // GELU preserves shape
};

