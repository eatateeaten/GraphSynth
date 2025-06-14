import { ModuleDef } from './types';

export const Linear: ModuleDef = {
    label: 'Linear',
    description: 'Applies a linear transformation to the incoming data',
    category: 'Linear',
    params: {
        input_features: {
            label: 'Input Features',
            description: 'Size of each input sample',
            type: 'number',
            default: 512,
            required: true
        },
        output_features: {
            label: 'Output Features',
            description: 'Size of each output sample',
            type: 'number',
            default: 256,
            required: true
        },
        bias: {
            label: 'Bias',
            description: 'Whether to add a learnable bias to the output',
            type: 'option',
            options: ['true', 'false'],
            default: 'true',
            required: false
        }
    },
    toPytorchExpr: (params) => `nn.Linear(${params.input_features}, ${params.output_features}, bias=${params.bias ?? true})`,
    shapeInference: (inShape, params) => {
        // TODO: Copy shape inference logic from torch file
        return [...inShape.slice(0, -1), params.output_features];
    }
};

export const Identity: ModuleDef = {
    label: 'Identity',
    description: 'A placeholder identity operator that is argument-insensitive',
    category: 'Linear',
    params: {},
    toPytorchExpr: () => 'nn.Identity()',
    shapeInference: (inShape) => inShape
};
