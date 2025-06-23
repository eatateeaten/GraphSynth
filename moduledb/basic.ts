import { ModuleDef } from './types';
import { shapeEqual } from './utils';
import { NodeType } from '../OpCompiler/types';

export const Tensor: ModuleDef = {
    label: 'Tensor',
    description: 'Represents input data or intermediate results',
    category: 'Basic',
    moduleType: NodeType.TENSOR,
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
    emitPytorchModule: undefined, 
    validateInputShape: (inShape, params) => {
        if(shapeEqual(inShape, params.shape))
            return [];
        return ["Input shape doesn't match with Tensor's shape"];
    },
    inferOutputShape: (_, params) => params.shape
};
