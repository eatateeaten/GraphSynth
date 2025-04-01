import { GraphNode } from './graph_node';
import { getDifferentiablePointWiseOpCode, getNonDifferentiablePointWiseOpCode } from './pointwise_op_map';


export abstract class MergeOp extends GraphNode {
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
        super(id, target);
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
    get prev(): GraphNode | null { return null; }
    set prev(node: GraphNode | null) { /* Do nothing */ }
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

    // addPrev 
    // Super(addPrev) 
    // main class addPrev should only take care of checkingIncomingShapeValidity. And this can be done for most Merge Operations 
    // For Reduceable Op, at any this stage they can compute an outShape (Reduceable Op's computeOutShape can be just an operation over the existing outShape)
    // However, For non-reduceable Op, they will have to check that they have filled all the requireed inputs before they can compute and outShape 

    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf: number, indexPrev?: number): void {
        // calling this from Graph.ts's connect 
        if (this._prevs[indexSelf] !== null && this._prevs[indexSelf] !== undefined) {
            throw new Error(`MergeOp already has a connection at input ${indexSelf}`); // a bit redundant 
        }
        //-------------------------------------------------------
        this.checkIncomingShapeMatch(prevOutShape); 
        if( this._numberOfMerges === this._prevs.filter(x => x != null).length + 1) {
            this.computeOutShape(); 
        }
        this._prevs[indexSelf] = prev;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("MergeOp already has a sink connection");
        }

        // Just set our next reference - Graph handles all validation and connections
        this._next = next;
    }

    deletePrev(indexSelf: number): void {
        //at this point we have already check that indexSelf is valid from the function calling deletePrev
        this._prevs[indexSelf] = null as unknown as GraphNode;
            // Just clear our reference and reset shapes 
        this._inShape[indexSelf] = null as unknown as number[]; 
        this._outShape = null; 
    }

    deleteNext(indexSelf?: number): void {
        // Just clear our next reference
        this._next = null;
    }
}

/**
 * PointwiseOp represents operations that take exactly two inputs with matching shapes
 * and perform element-wise operations between them.
 */
export class PointwiseOp extends MergeOp {
    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target, opType, params, 2); // Always 2 inputs for pointwise ops
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (!this._inShape.some(s => s !== null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShape.find(s => s !== null);
        if (!referenceShape) return;

        if (shape.length !== referenceShape.length) {
            throw new Error(`Shape rank mismatch: expected ${referenceShape.length}, got ${shape.length}`);
        }

        for (let i = 0; i < shape.length; i++) {
            if (shape[i] !== referenceShape[i]) {
                throw new Error(`Shape mismatch at dim ${i}: expected ${referenceShape[i]}, got ${shape[i]}`);
            }
        }
    }

    protected computeOutShape(): number[] {
        if (this._prevs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }

        const shape = this._prevs[0]?.outShape;
        if (!shape) {
            throw new Error("PointwiseOp requires first input to have defined shape");
        }

        return Array.isArray(shape) ? shape as number[] : [shape as number];
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }
        try {
            const diffOpCode = getDifferentiablePointWiseOpCode(this._opType, this._target);
            return `${inputs[0]} = ${diffOpCode}(${inputs[0]}, ${inputs[1]})`;
        } catch {
            const nonDiffOpCode = getNonDifferentiablePointWiseOpCode(this._opType, this._target);
            return `${inputs[0]} = ${nonDiffOpCode}(${inputs[0]}, ${inputs[1]})`;
        }
    }
}

/**
 * DotOp represents dot product operations between two tensors.
 * For 1D tensors: dot product
 * For 2D tensors: matrix multiplication
 * For higher dimensions: batched matrix multiplication
 */
export class DotOp extends MergeOp {
    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target, opType, params, 2); // Always 2 inputs for dot ops
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (!this._inShape.some(s => s !== null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShape.find(s => s !== null);
        if (!referenceShape) return;

        // For dot product, last dimension of first tensor must match first dimension of second tensor
        if (referenceShape[referenceShape.length - 1] !== shape[0]) {
            throw new Error(
                `Dot product dimension mismatch: last dim of first tensor (${referenceShape[referenceShape.length - 1]}) ` +
                `must match first dim of second tensor (${shape[0]})`
            );
        }

        // All other dimensions must match for batched operations
        for (let i = 0; i < Math.min(referenceShape.length - 1, shape.length - 1); i++) {
            if (referenceShape[i] !== shape[i]) {
                throw new Error(
                    `Batch dimension mismatch at dim ${i}: ` +
                    `expected ${referenceShape[i]}, got ${shape[i]}`
                );
            }
        }
    }

    protected computeOutShape(): number[] {
        if (this._prevs.length !== 2) {
            throw new Error("DotOp requires exactly 2 inputs");
        }

        const shape1 = this._prevs[0]?.outShape;
        const shape2 = this._prevs[1]?.outShape;
        if (!shape1 || !shape2) {
            throw new Error("DotOp requires both inputs to have defined shapes");
        }

        // For dot product, output shape is [batch_dims..., shape1[-2], shape2[-1]]
        const arr1 = Array.isArray(shape1) ? shape1 as number[] : [shape1 as number];
        const arr2 = Array.isArray(shape2) ? shape2 as number[] : [shape2 as number];
        return [...arr1.slice(0, -1), arr2[arr2.length - 1]];
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("DotOp requires exactly 2 inputs");
        }
        return `${inputs[0]} = torch.matmul(${inputs[0]}, ${inputs[1]})`;
    }
}

/**
 * CrossOp represents cross product operations between two tensors.
 * Only valid for 3D vectors (shape [..., 3]).
 */
export class CrossOp extends MergeOp {
    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target, opType, params, 2); // Always 2 inputs for cross ops
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (!this._inShape.some(s => s !== null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShape.find(s => s !== null);
        if (!referenceShape) return;

        // For cross product, last dimension must be 3
        if (shape[shape.length - 1] !== 3) {
            throw new Error(
                `Cross product requires 3D vectors, got shape [..., ${shape[shape.length - 1]}]`
            );
        }

        // All other dimensions must match for batched operations
        for (let i = 0; i < shape.length - 1; i++) {
            if (shape[i] !== referenceShape[i]) {
                throw new Error(
                    `Batch dimension mismatch at dim ${i}: ` +
                    `expected ${referenceShape[i]}, got ${shape[i]}`
                );
            }
        }
    }

    protected computeOutShape(): number[] {
        if (this._prevs.length !== 2) {
            throw new Error("CrossOp requires exactly 2 inputs");
        }

        const shape = this._prevs[0]?.outShape;
        if (!shape) {
            throw new Error("CrossOp requires first input to have defined shape");
        }

        // Cross product preserves the input shape
        return Array.isArray(shape) ? shape as number[] : [shape as number];
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("CrossOp requires exactly 2 inputs");
        }
        return `${inputs[0]} = torch.cross(${inputs[0]}, ${inputs[1]})`;
    }
}
