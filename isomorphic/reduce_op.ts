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
import { getElementwiseOpCode } from './torch_nn_module_op';
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
        /* sophia: implement this */
    }

    protected computeOutShape(): number[] {
        if (this._inShape.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 input tensors");
        }

        let resultShape = [...this._inShape[0]];

        for (let i = 1; i < this._inShape.length; i++) {
            const currentShape = this._inShape[i];
            
            if (currentShape.length !== resultShape.length) {
                throw new Error(`Input shapes must have the same rank. Shape at index ${i} has rank ${currentShape.length}, expected ${resultShape.length}`);
            }
            
            resultShape = resultShape.map((dim, j) => {
                const otherDim = currentShape[j];
                
                if (dim === otherDim) {
                    return dim;
                }
                if (dim === 1) {
                    return otherDim;
                }
                if (otherDim === 1) {
                    return dim;
                }
                throw new Error(
                    `Incompatible shapes for broadcasting at dimension ${j}: ` +
                    `${dim} and ${otherDim}. Dimensions must be equal or one must be 1.`
                );
            });
        }

        return resultShape;
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 inputs");
        }
        const opCode = getElementwiseOpCode(this._opType);
        
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
        /* sophia: implement this */
    }

    protected computeOutShape(): number[] {
        const dim = this._params.dim;
        if (dim < 0 || dim >= this._inShape[0].length) {
            throw new Error(`Invalid concatenation dimension ${dim} for input shape of length ${this._inShape[0].length}`);
        }

        const referenceShape = this._inShape[0];
        for (let i = 1; i < this._inShape.length; i++) {
            const shape = this._inShape[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For concatenation, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (j !== dim && shape[j] !== referenceShape[j]) {
                    throw new Error(`For concatenation, input shapes must match on all dimensions except the concatenation dimension. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        const outShape = [...referenceShape];
        outShape[dim] = this._inShape.reduce((sum, shape) => sum + shape[dim], 0);
        return outShape;
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }
}