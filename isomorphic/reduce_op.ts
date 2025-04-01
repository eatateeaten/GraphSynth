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
 * - to_torch_functional(): abstract in MergeOp, implemented in ReduceOp and its subclasses
 * - addPrev(): overridden in ReduceOp to handle dynamic input shapes
 */

import { MergeOp } from './merge_op';
import { getPointWiseReduceOpCode } from './torch_pointwise_reduce_op';
import { GraphNode } from './graph_node';

export abstract class ReduceOp extends MergeOp {
    protected _inShape: number[][];
    protected _outShape: number[]| null;
    public _prevs: GraphNode[] = [];
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;
    public _numberOfMerges: number; 

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}, 
        numberOfMerges: number 
    ) {
        super(id, target, opType, params, numberOfMerges);
        this._inShape = Array(numberOfMerges).fill(null)
        this._opType = opType;
        this._params = params;
        this._outShape = null; 
        this._numberOfMerges = numberOfMerges
    }
    
    protected abstract computeOutShape(): number[];
    protected abstract checkIncomingShapeMatch(shape: number[]): void; 
    abstract to_torch_functional(inputs: string[], outputs?: string[]): string;
    

    // Getters and setters
    get inShape(): number[][] { return this._inShape; }
    get outShape(): number[] | null { return this._outShape; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params as Record<string, any>) = { ...params };
        // Recalculate output shape
        this._outShape = this.computeOutShape();
    }

    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            throw new Error("MergeOp.addPrev requires an input index"); // a bit redundant if calling this from Graph.ts's connect 
        } 
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._inShape.length, "MergeOp.addPrev"); // a bit redundant if calling this from Graph.ts's connect 
        if (this._prevs[validatedIndex] !== null && this._prevs[validatedIndex] !== undefined) {
            throw new Error(`MergeOp already has a connection at input ${validatedIndex}`); // a bit redundant 
        }
        //-------------------------------------------------------
        this.checkIncomingShapeMatch(prevOutShape); 
        this._numberOfMerges
        this.computeOutShape; 
        this._prevs[validatedIndex] = prev;
    }
}

export class PointwiseReduce extends ReduceOp {
    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {},
        numberOfMerges: number
    ) {
        super(id, target, opType, params, numberOfMerges);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        // Skip check if this is the first shape
        const hasExistingShape = this._inShape.some(s => s !== null);
        if (!hasExistingShape) {
            return;
        }
        // Get reference shape (first non-null shape)
        const referenceShape = this._inShape.find(s => s !== null);
        if (!referenceShape) {
            return; // Shouldn't happen if hasExistingShape is true, but satisfies TypeScript
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

    protected computeOutShape(): number[] {
        // Find the first defined input shape
        const referenceShapeIndex = this._inShape.findIndex(s => s !== null);
        if (referenceShapeIndex === -1) {
            return []; // This part most likely cannot be reached? 
        }
        // Return a copy of the reference shape
        return [...this._inShape[referenceShapeIndex]];
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
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
    constructor(
        id: string,
        target: string,
        params: { dim: number },
        numberOfMerges: number 
    ) {
        super(id, target, "Concat", params, numberOfMerges);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        // Skip check if this is the first shape
        const hasExistingShape = this._inShape.some(s => s !== null);
        if (!hasExistingShape) {
            return;
        }
        
        // Get concat dimension from params
        const concatDim = this._params.dim;
        if (concatDim < 0 || concatDim >= shape.length) {
            throw new Error(
                `Invalid concatenation dimension ${concatDim} for input shape of length ${shape.length}`
            );
        }
        
        // Get reference shape (first non-null shape)
        const referenceShape = this._inShape.find(s => s !== null);
        if (!referenceShape) {
            return; // Shouldn't happen if hasExistingShape is true, but satisfies TypeScript
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

    protected computeOutShape(): number[] {
        // Find the first defined input shape
        const referenceShapeIndex = this._inShape.findIndex(s => s !== null);
        if (referenceShapeIndex === -1) {
            return []; // No shapes yet
        }
        
        const referenceShape = this._inShape[referenceShapeIndex];
        const dim = this._params.dim;
        
        // Create the output shape as a copy of the reference shape
        const outShape = [...referenceShape];
        
        // Sum up the sizes along the concatenation dimension
        outShape[dim] = this._inShape.reduce((sum, shape) => 
            shape ? sum + shape[dim] : sum, 0
        );
        
        return outShape;
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }
}