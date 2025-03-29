import { Shape } from './shape';
import { CheckerNode, NodeParams, OutputError } from './node';
import { NodeMetadata } from './node';

export interface TensorParams extends NodeParams {
    shape: Shape;
}

export class Tensor extends CheckerNode<TensorParams> {
    static readonly type = 'tensor' as const;
    static readonly description = 'Creates a tensor with the given shape.';

    static getMeta(): NodeMetadata<TensorParams> {
        return {
            label: 'Tensor',
            description: this.description,
            category: 'basic',
            paramFields: {
                shape: {
                    label: 'Shape',
                    description: 'Dimensions of the tensor',
                    type: 'shape',
                    default: [1, 64, 32, 32]
                }
            }
        } as const;
    }

    static validateParams(params: NodeParams): string | null {
        if (!params || typeof params !== 'object') return 'Invalid params';
        const p = params as TensorParams;
        
        if (!p.shape) return 'Shape must be specified';
        if (!Array.isArray(p.shape)) return 'Shape must be an array';
        if (p.shape.length === 0) return 'Shape cannot be empty';
        
        return null;
    }

    constructor(params: TensorParams) {
        super(params);
        this.in_shape = params.shape;
        this.updateOutShape();
    }

    computeOutShape(in_shape: Shape): Shape {
        return in_shape;
    }

    setInShape(shape: Shape | null): void {
        if (shape !== null && !Shape.equals(shape, this.in_shape)) {
            throw new OutputError("Cannot change tensor shape: tensor shapes are immutable");
        }
    }
}

export interface ReshapeParams extends NodeParams {
    out_dim: Shape;
}

export class Reshape extends CheckerNode<ReshapeParams> {
    static readonly type = 'reshape' as const;
    static readonly description = 'Reshapes the input tensor to the specified dimensions.';

    static getMeta(): NodeMetadata<ReshapeParams> {
        return {
            label: 'Reshape',
            description: this.description,
            category: 'basic',
            paramFields: {
                out_dim: {
                    label: 'Output Dimensions',
                    description: 'Target shape (use -1 for automatic inference)',
                    type: 'shape',
                    allowNegativeOne: true,
                    default: [1, -1]
                }
            }
        } as const;
    }

    static validateParams(params: NodeParams): string | null {
        if (!params || typeof params !== 'object') return 'Invalid params';
        const p = params as ReshapeParams;

        if (!p.out_dim) return 'Output dimensions must be specified';
        if (!Array.isArray(p.out_dim)) return 'Output dimensions must be an array';
        if (p.out_dim.length === 0) return 'Output dimensions cannot be empty';

        try {
            Shape.validateReshapeShape(p.out_dim);
            return null;
        } catch (e) {
            return e instanceof Error ? e.message : String(e);
        }
    }

    computeOutShape(in_shape: Shape): Shape {
        const out_dim = this.params.out_dim;
        const total_in = in_shape.reduce((a, b) => a * b, 1);
        
        if (out_dim.includes(-1)) {
            const known_out = out_dim.filter(d => d !== -1).reduce((a, b) => a * b, 1);
            const missing_dim = total_in / known_out;
            if (!Number.isInteger(missing_dim)) {
                throw new OutputError("Cannot compute missing dimension: total elements do not match");
            }
            return out_dim.map(d => d === -1 ? missing_dim : d);
        }

        const total_out = out_dim.reduce((a, b) => a * b, 1);
        if (total_in !== total_out) {
            throw new OutputError(`Cannot reshape from ${in_shape} to ${out_dim}: total elements do not match`);
        }
        return out_dim;
    }
}

export const UtilNodes = {
    [Tensor.type]: Tensor,
    [Reshape.type]: Reshape,
} as const;

export type UtilNodeParams = {
    [Tensor.type]: TensorParams;
    [Reshape.type]: ReshapeParams;
};
