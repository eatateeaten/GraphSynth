import { GraphNode } from './graph';
import { getElementwiseOpCode, forwardShapeInference } from './torch_nn_module_op';
import { Tensor } from './tensor';
import { BranchOp } from './branch_op';
import { MergeOp } from './merge_op';

export class Op extends GraphNode {
    protected _inShape: number[] | null = null;
    protected _outShape: number[] | null = null;
    protected _prev: GraphNode | null = null;
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target);
        this._opType = opType;
        this._params = params;
    }

    protected computeOutShape(): number[] {
        if (!this._inShape) {
            throw new Error(`Cannot compute output shape without input shape for operation ${this._opType}`);
        }
        
        // Use forwardShapeInference for torch operations
        if (this._target === "torch") {
            try {
                return forwardShapeInference(this._opType, this._inShape, this._params);
            } catch (err: any) {
                throw new Error(`Shape inference error for ${this._opType}: ${err.message}. Consider using a different set of parameters.`);
            }
        }
        
        // For non-torch operations, throw an error
        throw new Error(`No shape inference implementation available for target '${this._target}' and operation '${this._opType}'`);
    }

    to_torch_functional(inputs: string[]): string {
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        
        if (this._inShape === null || this._outShape === null) {
            throw new Error("Cannot generate torch code: operation has undefined input or output shape");
        }
        
        return `${inputs[0]} = torch.${this._opType}(${inputs[0]})`;
    }

    // Getters and setters
    get inShape(): number[] | null { return this._inShape; }
    set inShape(shape: number[] | null) { 
        // inShape can only be set during connection
        throw new Error("Cannot directly set inShape for Op. Connect a source node instead.");
    }
    get outShape(): number[] | null { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("Op already has a source connection");
        }
        
        // Get Prev's outShape 
        let prevOutShape: number[];
        if (prev instanceof Tensor || prev instanceof Op || prev instanceof MergeOp) {
            if (prev.outShape === null) {
                throw new Error(`Cannot connect to ${prev.constructor.name} with id ${prev.id}: output shape is undefined`);
            }
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            if (indexPrev !== undefined) {
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "Op.connectSource (BranchOp output)");
                const branchOutShape = prev.outShape[validatedPrevIndex];
                if (branchOutShape === null) {
                    throw new Error(`Cannot connect to BranchOp with id ${prev.id} at output ${validatedPrevIndex}: output shape is undefined`);
                }
                prevOutShape = branchOutShape;
                indexPrev = validatedPrevIndex;
            } else {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
        } else {
            throw new Error(`Cannot connect to node of type ${prev.constructor.name}`);
        }
        
        // Set inShape based on the previous node's outShape
        this._inShape = [...prevOutShape];
        
        // Compute outShape based on the new inShape
        try {
            this._outShape = this.computeOutShape();
        } catch (err: any) {
            // Reset inShape if shape inference fails
            this._inShape = null;
            throw err;
        }
        
        // Make the bidirectional connection
        this._prev = prev;
        if (prev instanceof BranchOp && indexPrev !== undefined) {
            (prev as BranchOp)._nexts[indexPrev] = this;
        } else {
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("Op already has a sink connection");
        }
        
        // Get Next's inShape 
        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op || next instanceof BranchOp) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof MergeOp) {
            if (indexNext === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            const validatedNextIndex = GraphNode.validateIndex(indexNext, next.inShape.length, "Op.connectSink (MergeOp input)");
            nextInShape = next.inShape[validatedNextIndex] as number[];
            indexNext = validatedNextIndex;
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }
        
        // Ensure we have output shape
        if (this._outShape === null) {
            throw new Error("Cannot connect sink: Op has no input connection to determine output shape");
        }
        
        // Do a shape match check 
        if (!GraphNode.shapeMatch(this._outShape as number[], nextInShape)) {
            throw new Error(`Shape mismatch: Cannot connect Op with output shape [${this._outShape}] to next node with input shape [${nextInShape}]`);
        }
        
        // Make the bidirectional connection 
        this._next = next;
        if (next instanceof MergeOp && indexNext !== undefined) {
            (next as MergeOp)._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(indexSelf?: number): void {
        if (this._prev) {
            if (this._prev instanceof BranchOp) {  //find itself's reference and disconnect 
                const branchIndex = (this._prev as BranchOp)._nexts.indexOf(this);
                if (branchIndex >= 0) {
                    (this._prev as BranchOp)._nexts[branchIndex] = null as unknown as GraphNode;
                }
            } else {
                this._prev.next = null;
            }
            this._prev = null;
        }
    }

    disconnectSink(indexSelf?: number): void {
        if (this._next) {
            if (this._next instanceof MergeOp) {   //find itself's reference and disconnect 
                const mergeIndex = (this._next as MergeOp)._prevs.indexOf(this);
                if (mergeIndex >= 0) {
                    (this._next as MergeOp)._prevs[mergeIndex] = null as unknown as GraphNode;
                }
            } else {
                this._next.prev = null;
            }
            this._next = null;
        }
    }
} 