import { ModuleDef } from './types';

export const Tensor: ModuleDef = {
    label: 'Tensor',
    description: 'Represents input data or intermediate results',
    category: 'Basic',
    params: {
        shape: {
            label: 'Shape',
            description: 'Dimensions of the tensor (comma-separated)',
            type: 'shape',
            default: [3, 64, 64],
            allowNegativeOne: false,
            required: true
        },
        variableName: {
            label: 'Variable Name',
            description: 'Name of this tensor',
            type: 'string',
            default: "tensor0",
            required: true
        }
    },
    toPytorchExpr: () => '', // Tensors don't generate code
    shapeInference: (_, params) => params.shape
}; 