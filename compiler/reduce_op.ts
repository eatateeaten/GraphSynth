/**
 * Class Diagram:
 * 
 * GraphNode (base)
 *    ↑
 * MergeOp (abstract)
 *    ↑
 * ReduceOp (base)
 *    ↑
 * ├── PointwiseReduce
 * └── Concat
 * 
 * Key relationships:
 * - ReduceOp extends MergeOp
 * - PointwiseReduce and Concat extend ReduceOp
 * - All classes inherit from GraphNode
 * 
 * Key methods:
 * - computeOutShape(): abstract in MergeOp, implemented in ReduceOp and its subclasses
 * - emitTorchFunctional(): abstract in MergeOp, implemented in ReduceOp and its subclasses
 * - addPrev(): overridden in ReduceOp to handle dynamic input shapes
 */

import { MergeOp } from './merge_op';
import { getPointWiseReduceOpCode } from './torch_pointwise_reduce_op';
import { assert } from './utils';

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
    abstract emitTorchFunctional(inputs: string[], outputs?: string[]): string;
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

    emitTorchFunctional(inputs: string[], outputs?: string[]): string {
        if (inputs.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 inputs");
        }
        const opCode = getPointWiseReduceOpCode(this._opType);
        
        const result = inputs.reduce((acc, curr) => 
            acc ? `${opCode}(${acc}, ${curr})` : curr
        );
        
        return `${inputs[0]} = ${result}`;
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

    set params(params: Record<string, any>) {
        assert(params.dim, "Dimension is required for Concat");
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

    emitTorchFunctional(inputs: string[], outputs?: string[]): string {
        const outVar = outputs && outputs.length > 0 ? outputs[0] : inputs[0];
        return `${outVar} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }
}