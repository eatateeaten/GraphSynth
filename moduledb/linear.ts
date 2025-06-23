import { ModuleDef } from './types';

export const Linear: ModuleDef = {
    label: 'Linear',
    description: 'Applies a linear transformation to the incoming data',
    category: 'Linear',
    moduleType: "Op",
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
            type: 'boolean',
            default: true,
            required: false
        }
    },
    emitPytorchModule: (params) => `nn.Linear(${params.input_features}, ${params.output_features}, bias=${params.bias ?? true})`,
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        
        // Linear requires at least 1D input
        if (inShape.length < 1) {
            errors.push("Linear requires at least 1D input");
        }
        
        // Last dimension must be equal to input features
        if (inShape.length >= 1 && inShape[inShape.length - 1] !== params.input_features) {
            errors.push(`Last dimension must be equal to input features: expected ${params.input_features}, got ${inShape[inShape.length - 1]}`);
        }
        
        return errors;
    },
    inferOutputShape: (inShape, params) => {
        /* From torch documentation:
         * Input: (∗,Hin) where ∗ means any number of dimensions including none and Hin = in_features
         * Output: (∗,Hout) where all but the last dimension are the same shape as the input and Hout = out_features */
        return [...inShape.slice(0, -1), params.output_features];
    }
};

export const Identity: ModuleDef = {
    label: 'Identity',
    description: 'A placeholder identity operator that is argument-insensitive',
    category: 'Linear',
    moduleType: "Op",
    params: {},
    emitPytorchModule: () => 'nn.Identity()',
    validateInputShape: (_inShape) => [],
    inferOutputShape: (inShape) => inShape
};
