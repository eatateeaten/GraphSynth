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

// TODO: Add Sigmoid, Tanh, LeakyReLU, ELU, SELU, Swish, GELU, etc.

