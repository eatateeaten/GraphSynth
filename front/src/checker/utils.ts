// src/checker/utils.ts
import { Shape } from './shape';
import { CheckerNode, NodeParams } from './node';

export interface TensorParams extends NodeParams {
    shape: Shape;
}

export class Tensor extends CheckerNode<TensorParams> {
    static readonly type = 'tensor' as const;

    constructor(params: TensorParams) {
        super(params);
        this.in_shape = params.shape;
        this.recompute_out_shape();
    }

    validate_params(): void {
        const { shape } = this.params;
        if (!Array.isArray(shape)) {
            throw new Error("shape must be an array");
        }
    }

    compute_out_shape(in_shape: Shape): Shape {
        return in_shape;
    }

    set_in_shape(shape: Shape | null): void {
        if (shape !== null && !shape.equals(this.in_shape)) {
            throw new Error("Cannot change tensor shape: tensor shapes are immutable");
        }
    }
}

export interface ReshapeParams extends NodeParams {
    out_dim: Shape;
}

export class Reshape extends CheckerNode<ReshapeParams> {
    static readonly type = 'reshape' as const;

    constructor(params: ReshapeParams) {
        super(params);
    }

    validate_params(): void {
        const { out_dim } = this.params;
        if (!Array.isArray(out_dim)) {
            throw new Error("out_dim must be an array");
        }

        const inferredDims = out_dim.filter(d => d === -1);
        if (inferredDims.length > 1) {
            throw new Error("Only one dimension can be inferred (-1) in out_dim");
        }

        const invalidDims = out_dim.filter(d => d !== -1 && d <= 0);
        if (invalidDims.length > 0) {
            throw new Error(`Invalid dimension ${invalidDims[0]} in out_dim: must be positive or -1`);
        }
    }

    compute_out_shape(in_shape: Shape): Shape {
        const out_dim = this.params.out_dim;
        const total_in = in_shape.reduce((a, b) => a * b, 1);
        
        if (out_dim.includes(-1)) {
            const known_out = out_dim.filter(d => d !== -1).reduce((a, b) => a * b, 1);
            const missing_dim = total_in / known_out;
            if (!Number.isInteger(missing_dim)) {
                throw new Error("Cannot compute missing dimension: total elements do not match");
            }
            return new Shape(out_dim.map(d => d === -1 ? missing_dim : d));
        }

        const total_out = out_dim.reduce((a, b) => a * b, 1);
        if (total_in !== total_out) {
            throw new Error(`Cannot reshape from ${in_shape} to ${out_dim}: total elements do not match`);
        }
        return out_dim;
    }
}

export type UtilNodeParams = {
    [Tensor.type]: TensorParams;
    [Reshape.type]: ReshapeParams;
};
