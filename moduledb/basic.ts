import { ModuleDef } from './types';
import { shapeEqual } from './utils';

export const Tensor: ModuleDef = {
    label: 'Tensor',
    description: 'Represents input data or intermediate results',
    category: 'Basic',
    moduleType: "Tensor",
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
    emitPytorchModule: () => '', // Tensors don't generate code
    validateInputShape: (inShape, params) => {
        if(shapeEqual(inShape, params.shape))
            return [];
        return ["Input shape doesn't match with Tensor's shape"];
    },
    inferOutputShape: (_, params) => params.shape
};
