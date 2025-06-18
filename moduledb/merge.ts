import { ModuleDef } from './types';

// TODO: Add actual PyTorch merge/combine operations here, such as:
// - torch.cat (concatenation) 
// - torch.stack (stacking)
// - Element-wise operations that can be expressed as PyTorch modules
// 
// Note: Graph-level operations like PointwiseOp, DotOp, CrossOp belong in 
// OpCompiler/merge_op.ts since they handle multiple inputs and graph topology

// Helper functions for pointwise operations
function getDifferentiablePointWiseOpCode(opType: string): string {
    switch (opType.toLowerCase()) {
        case 'add': return 'torch.add';
        case 'mul': return 'torch.mul';
        case 'div': return 'torch.div';
        case 'pow': return 'torch.pow';
        default: throw new Error(`Unknown differentiable pointwise operation: ${opType}`);
    }
}

function getNonDifferentiablePointWiseOpCode(opType: string): string {
    switch (opType.toLowerCase()) {
        case 'min': return 'torch.min';
        case 'max': return 'torch.max';
        case 'and': return 'torch.logical_and';
        case 'or': return 'torch.logical_or';
        case 'eq': return 'torch.eq';
        case 'ne': return 'torch.ne';
        case 'lt': return 'torch.lt';
        case 'le': return 'torch.le';
        case 'gt': return 'torch.gt';
        case 'ge': return 'torch.ge';
        default: throw new Error(`Unknown non-differentiable pointwise operation: ${opType}`);
    }
}

export const PointwiseOp: ModuleDef = {
    label: 'PointwiseOp',
    description: 'Element-wise operations between two tensors with matching shapes',
    category: 'Math',
    moduleType: 'PointwiseOp',
    params: {
        opType: {
            label: 'Operation',
            description: 'Type of pointwise operation to perform',
            type: 'option',
            default: 'add',
            options: ['add', 'mul', 'div', 'pow', 'min', 'max', 'and', 'or', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'],
            required: true
        }
    },
    toPytorchModule: (params) => {
        const opType = params.opType || 'add';
        try {
            return getDifferentiablePointWiseOpCode(opType);
        } catch {
            return getNonDifferentiablePointWiseOpCode(opType);
        }
    },
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        if (!inShape || inShape.length === 0) {
            errors.push('Input shape is required');
        }
        if (!params.opType) {
            errors.push('Operation type is required');
        }
        return errors;
    },
    inferOutputShape: (inShape, params) => {
        return [...inShape]; // Pointwise operations preserve input shape
    }
};

export const DotOp: ModuleDef = {
    label: 'DotOp',
    description: 'Dot product (matrix multiplication) between two tensors',
    category: 'Math',
    moduleType: 'DotOp',
    params: {},
    toPytorchModule: (params) => {
        return 'torch.matmul';
    },
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        if (!inShape || inShape.length === 0) {
            errors.push('Input shape is required');
        }
        return errors;
    },
    inferOutputShape: (inShape, params) => {
        // For dot product, this is complex as it depends on both input shapes
        // This is a simplified version - real implementation would need both shapes
        return inShape; // Placeholder - actual implementation needs both tensor shapes
    }
};

export const CrossOp: ModuleDef = {
    label: 'CrossOp',
    description: 'Cross product of two 3-dimensional vectors',
    category: 'Math',
    moduleType: 'CrossOp',
    params: {},
    toPytorchModule: (params) => {
        return 'torch.cross';
    },
    validateInputShape: (inShape, params) => {
        const errors: string[] = [];
        if (!inShape || inShape.length === 0) {
            errors.push('Input shape is required');
        } else if (inShape[inShape.length - 1] !== 3) {
            errors.push('Cross product requires 3D vectors (last dimension must be 3)');
        }
        return errors;
    },
    inferOutputShape: (inShape, params) => {
        return [...inShape]; // Cross product preserves input shape
    }
};
