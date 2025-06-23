import { ModuleDef } from './types';

export const reshapeModules: Record<string, ModuleDef> = {
    'Reshape': {
        label: 'Reshape',
        description: 'Reshapes the input tensor to the specified shape',
        category: 'Reshape',
        moduleType: 'Op',
        params: {
            shape: {
                label: 'Shape',
                description: 'The desired shape (comma-separated numbers)',
                type: 'shape',
                allowNegativeOne: true,
                required: true
            }
        },
        emitPytorchModule: (params) => `torch.reshape(${params['shape'].join(', ')})`,
        validateInputShape: (inShape, params) => {
            const targetShape = params['shape'];
            if (!Array.isArray(targetShape)) {
                return [`Reshape shape parameter must be an array, got ${targetShape}`];
            }
            
            const hasNegOne = targetShape.includes(-1);
            const inElements = inShape.reduce((acc, dim) => acc * dim, 1);
            
            if (!hasNegOne) {
                const outElements = targetShape.reduce((acc, dim) => acc * dim, 1);
                if (inElements !== outElements) {
                    return [`Reshape total elements mismatch: input has ${inElements} elements, but target shape has ${outElements} elements`];
                }
            } else {
                const negOnes = targetShape.filter((d: number) => d === -1).length;
                if (negOnes !== 1) {
                    return [`Reshape shape can have at most one -1 dimension, got ${negOnes}`];
                }
                
                const specifiedElements = targetShape.filter((d: number) => d !== -1).reduce((acc: number, dim: number) => acc * dim, 1);
                if (inElements % specifiedElements !== 0) {
                    return [`Reshape cannot infer size for -1 dimension: input has ${inElements} elements, which is not divisible by product of specified dimensions (${specifiedElements})`];
                }
            }
            
            return [];
        },
        inferOutputShape: (inShape, params) => {
            const targetShape = params['shape'];
            const hasNegOne = targetShape.includes(-1);
            const inElements = inShape.reduce((acc, dim) => acc * dim, 1);
            
            if (!hasNegOne) {
                return [...targetShape];
            } else {
                const specifiedElements = targetShape.filter((d: number) => d !== -1).reduce((acc: number, dim: number) => acc * dim, 1);
                const negOneDimValue = inElements / specifiedElements;
                const outputShape = targetShape.map((d: number) => d === -1 ? negOneDimValue : d);
                return outputShape;
            }
        }
    },

    'Permute': {
        label: 'Permute',
        description: 'Permutes the dimensions of the input tensor',
        category: 'Reshape',
        moduleType: 'Op',
        params: {
            dims: {
                label: 'Dimensions',
                description: 'The desired ordering of dimensions (comma-separated numbers)',
                type: 'shape',
                allowNegativeOne: false,
                required: true
            }
        },
        emitPytorchModule: (params) => `torch.permute(${params['dims'].join(', ')})`,
        validateInputShape: (inShape, params) => {
            const dims = params['dims'];
            if (!Array.isArray(dims)) {
                return [`Permute dims parameter must be an array, got ${dims}`];
            }
            if (dims.length !== inShape.length) {
                return [`Permute dims must have same length as input shape, got ${dims.length} vs ${inShape.length}`];
            }
            
            const sorted = [...dims].sort((a, b) => a - b);
            for (let i = 0; i < sorted.length; i++) {
                if (sorted[i] !== i) {
                    return [`Permute dims must contain all dimensions from 0 to ${inShape.length - 1} exactly once`];
                }
            }
            
            return [];
        },
        inferOutputShape: (inShape, params) => {
            const dims = params['dims'];
            return dims.map((d: number) => inShape[d]);
        }
    },

    'Flatten': {
        label: 'Flatten',
        description: 'Flattens input by reshaping it into a one-dimensional tensor',
        category: 'Reshape',
        moduleType: 'Op',
        params: {
            start_dim: {
                label: 'Start Dimension',
                description: 'First dim to flatten',
                type: 'number',
                default: 1,
                required: false
            },
            end_dim: {
                label: 'End Dimension',
                description: 'Last dim to flatten',
                type: 'number',
                default: -1,
                required: false
            }
        },
        emitPytorchModule: (params) => `nn.Flatten(start_dim=${params['start_dim'] ?? 1}, end_dim=${params['end_dim'] ?? -1})`,
        validateInputShape: (inShape, params) => {
            const start_dim = params['start_dim'] ?? 1;
            let end_dim = params['end_dim'] ?? -1;
            
            if (end_dim < 0) {
                end_dim = inShape.length + end_dim;
            }
            
            if (start_dim < 0 || start_dim >= inShape.length) {
                return [`Flatten invalid start_dim=${start_dim} for input shape ${inShape}`];
            }
            
            if (end_dim < start_dim || end_dim >= inShape.length) {
                return [`Flatten invalid end_dim=${end_dim} for input shape ${inShape}`];
            }
            
            return [];
        },
        inferOutputShape: (inShape, params) => {
            const start_dim = params['start_dim'] ?? 1;
            let end_dim = params['end_dim'] ?? -1;
            
            if (end_dim < 0) {
                end_dim = inShape.length + end_dim;
            }
            
            let flattenedSize = 1;
            for (let i = start_dim; i <= end_dim; i++) {
                flattenedSize *= inShape[i];
            }
            
            const outShape = [
                ...inShape.slice(0, start_dim),
                flattenedSize,
                ...inShape.slice(end_dim + 1)
            ];
            
            return outShape;
        }
    },

    'Unflatten': {
        label: 'Unflatten',
        description: 'Unflattens a tensor dim expanding it to a desired shape',
        category: 'Reshape',
        moduleType: 'Op',
        params: {
            dim: {
                label: 'Dimension',
                description: 'Dimension to unflatten',
                type: 'number',
                default: 1,
                required: true
            },
            unflattened_size: {
                label: 'Unflattened Size',
                description: 'New shape of the unflattened dimension',
                type: 'shape',
                allowNegativeOne: false,
                required: true
            }
        },
        emitPytorchModule: (params) => `nn.Unflatten(dim=${params['dim']}, unflattened_size=${params['unflattened_size']})`,
        validateInputShape: (inShape, params) => {
            const dim = params['dim'];
            const unflattened_size = params['unflattened_size'];
            
            if (dim < 0 || dim >= inShape.length) {
                return [`Unflatten invalid dim=${dim} for input shape ${inShape}`];
            }
            
            if (!Array.isArray(unflattened_size)) {
                return [`Unflatten unflattened_size must be an array, got ${unflattened_size}`];
            }
            
            const targetElements = unflattened_size.reduce((acc, d) => acc * d, 1);
            if (inShape[dim] !== targetElements) {
                return [`Unflatten dimension ${dim} has size ${inShape[dim]} but unflattened_size requires ${targetElements} elements`];
            }
            
            return [];
        },
        inferOutputShape: (inShape, params) => {
            const outShape = [...inShape];
            const dim = params['dim'];
            const unflattened_size = params['unflattened_size'];
            outShape.splice(dim, 1, ...unflattened_size);
            return outShape;
        }
    }
};
