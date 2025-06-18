import { MergeOp } from './merge_op';
import { ParamError } from './types';

/* huh. it's interesting. reduceop doesn't do anything different than mergeop. keeping it for semantic reasons */
export abstract class ReduceOp extends MergeOp {
    constructor(
        id: string,
        opType: string,
        numberOfMerges: number,
        params: Record<string, any> = {}, 
    ) {
        super(id, opType, numberOfMerges, params);
    }

    protected abstract computeOutShape(): number[] | null;
    protected abstract checkIncomingShapeMatch(shape: number[]): void; 
    abstract toTorchModule(): string;
    abstract toIR(): string;
}

export class PointwiseReduce extends ReduceOp {
    constructor(
        id: string,
        opType: string,
        numberOfMerges: number,
        params: Record<string, any> = {},
    ) {
        super(id, opType, numberOfMerges, params);
    }

    static fromParams(id: string, params: Record<string, any>): PointwiseReduce {
        if (!params.opType)
            throw new ParamError("Operation type is required for PointwiseReduce");
        if (!params.numberOfMerges || params.numberOfMerges < 2)
            throw new ParamError("NumberOfMerges must be at least 2 for PointwiseReduce");
        return new PointwiseReduce(id, params.opType, params.numberOfMerges, params);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        // Get reference shape (first non-null shape)
        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) {
            return;
        }

        // Validate rank (number of dimensions)
        if (shape.length !== referenceShape.length) {
            throw new Error(
                `Shape mismatch: expected rank ${referenceShape.length}, got ${shape.length}`
            );
        }

        // Validate each dimension
        const mismatchedDimension = shape.findIndex((dim, i) => dim !== referenceShape[i]);
        if (mismatchedDimension !== -1) {
            throw new Error(
                `Shape mismatch at dimension ${mismatchedDimension}: ` + 
                `expected ${referenceShape[mismatchedDimension]}, got ${shape[mismatchedDimension]}`
            );
        }
    }

    protected computeOutShape(): number[] | null {
        // Find the first defined input shape
        const referenceShapeIndex = this._inShapes.findIndex(s => s !== null);
        if (referenceShapeIndex === -1)
            return null;

        // Return a copy of the reference shape
        return [...this._inShapes[referenceShapeIndex]!];
    }

    toTorchModule(): string {
        // TODO: fix this
        return "toTorchModule for PointwiseReduce not implemented";
    }

    toIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `${this._opType}Reduce(inputs=${this._numberOfMerges}) -> ${shapeStr}`;
    }
}

export class Concat extends ReduceOp {
    private _dim: number;

    constructor(
        id: string,
        dim: number,
        numberOfMerges: number,
        params: Record<string, any>
    ) {
        super(id, "Concat", numberOfMerges, params);
        this._dim = dim;
    }

    /** Validate params and construct if OK */
    static fromParams(id: string, params: Record<string, any>): Concat {
        if (params.dim === undefined)
            throw new ParamError("Dimension is required for Concat");
        if (!params.numberOfMerges || params.numberOfMerges < 2)
            throw new ParamError("NumberOfMerges must be at least 2 for Concat");
        return new Concat(id, params.dim, params.numberOfMerges, params);
    }

    set params(params: Record<string, any>) {
        if (params.dim === undefined)
            throw new ParamError("Dimension is required for Concat");
        super.params = params;
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        // Get reference shape (first non-null shape)
        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) {
            return; // Shouldn't happen if hasExistingShape is true, but satisfies TypeScript
        }

        // Get concat dimension from params
        const concatDim = this._dim;
        if (concatDim < 0 || concatDim >= shape.length) {
            throw new Error(
                `Invalid concatenation dimension ${concatDim} for input shape of length ${shape.length}`
            );
        }

        // Validate rank (number of dimensions)
        if (shape.length !== referenceShape.length) {
            throw new Error(
                `Rank mismatch: expected rank ${referenceShape.length}, got ${shape.length}`
            );
        }

        // Validate each dimension except concat dimension
        for (let i = 0; i < shape.length; i++) {
            if (i !== concatDim && shape[i] !== referenceShape[i]) {
                throw new Error(
                    `Shape mismatch at dimension ${i}: expected ${referenceShape[i]}, got ${shape[i]}`
                );
            }
        }
    }

    protected computeOutShape(): number[] | null {
        // Find the first defined input shape
        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) {
            return null; // No shapes yet
        }

        // Validate concatenation dimension
        const dim = this._dim;
        if (dim === undefined || dim < 0 || dim >= referenceShape.length) {
            throw new Error(`Invalid concatenation dimension ${dim} for shape of length ${referenceShape.length}`);
        }

        // Create the output shape as a copy of the reference shape
        const outShape = [...referenceShape];

        // Sum up the sizes along the concatenation dimension
        let totalSize = 0;
        for (const shape of this._inShapes) {
            if (!shape) continue;

            // Validate shape compatibility
            if (shape.length !== referenceShape.length) {
                throw new Error(`Shape rank mismatch: expected ${referenceShape.length}, got ${shape.length}`);
            }

            // Check all dimensions except concat dimension match
            for (let i = 0; i < shape.length; i++) {
                if (i !== dim && shape[i] !== referenceShape[i]) {
                    throw new Error(`Shape mismatch at dimension ${i}: expected ${referenceShape[i]}, got ${shape[i]}`);
                }
            }

            totalSize += shape[dim];
        }

        outShape[dim] = totalSize;
        return outShape;
    }

    toTorchModule(): string {
        return "toTorchModule for Concat not implemented";
    }

    toIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `Concat(dim=${this._dim}, inputs=${this._numberOfMerges}) -> ${shapeStr}`;
    }
}
