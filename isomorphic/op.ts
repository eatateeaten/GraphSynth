import { GraphNode } from './graph';
import { getElementwiseOpCode } from './torch_nn_module_op';
import { Tensor } from './tensor';
import { BranchOp } from './branch_op';
import { MergeOp } from './merge_op';

export class Op extends GraphNode {
    protected _inShape: number[];
    protected _outShape: number[];
    protected _prev: GraphNode | null = null;
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;

    constructor(
        id: string,
        inShape: number[],
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target);
        this._inShape = inShape;
        this._opType = opType;
        this._params = params;
        this._outShape = this.computeOutShape();
    }

    protected computeOutShape(): number[] {
        // Most unary ops preserve shape
        return [...this._inShape];
    }

    to_torch_functional(inputs: string[]): string {
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        return `${inputs[0]} = torch.${this._opType}(${inputs[0]})`;
    }

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[] { return this._outShape; }
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
        //Get Prev's outShape 
        let prevOutShape: number[];
        if (prev instanceof Tensor || prev instanceof Op || prev instanceof MergeOp) {
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            if (indexPrev !== undefined) {
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "Op.connectSource (BranchOp output)");
                prevOutShape = prev.outShape[validatedPrevIndex] as number[];
                indexPrev = validatedPrevIndex;
            } else {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
        } else {
            throw new Error(`Cannot connect to node of type ${prev.constructor.name}`);
        }
        
        //Do a shapeMatch check 
        if (!GraphNode.shapeMatch(this._inShape, prevOutShape)) {
            throw new Error(`Shape mismatch: Cannot connect Op with input shape [${this._inShape}] to previous node with output shape [${prevOutShape}]`);
        }
        
        //Make the bidirectional connection 
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
        //Get Next's inShape 
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
        //Do a shapeMatch check 
        if (!GraphNode.shapeMatch(this._outShape, nextInShape)) {
            throw new Error(`Shape mismatch: Cannot connect Op with output shape [${this._outShape}] to next node with input shape [${nextInShape}]`);
        }
        //Make the bidirectional connection 
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