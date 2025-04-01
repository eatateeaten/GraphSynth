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
export abstract class PointwiseOp extends MergeOp {
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

        const diffOpCode = getDifferentiablePointWiseOpCode(this._opType, this._target);
        const nonDiffOpCode = getNonDifferentiablePointWiseOpCode(this._opType, this._target);
        return `${inputs[0]} = ${diffOpCode}(${inputs[0]}, ${inputs[1]})`;
    }
}
