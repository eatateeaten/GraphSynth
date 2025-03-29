import { GraphNode } from './graph';
import { BranchOp } from './branch_op';
import { Op } from './op';
import { MergeOp } from './merge_op';

export class Tensor extends GraphNode {
    protected _inShape: number[];
    protected _outShape: number[];
    protected _prev: GraphNode | null = null;
    protected _next: GraphNode | null = null;

    constructor(id: string, shape: number[], target: string) {
        super(id, target);
        this._inShape = shape;
        this._outShape = shape;
    }

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[] { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("Tensor already has a source connection");
        }

        let prevOutShape: number[];
        if (prev instanceof Tensor || prev instanceof Op) {
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            if (indexPrev !== undefined) {
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "Tensor.connectSource (BranchOp output)");
                prevOutShape = prev.outShape[validatedPrevIndex] as number[];
                indexPrev = validatedPrevIndex;
            } else {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
        } else if (prev instanceof MergeOp) {
            prevOutShape = prev.outShape;
        } else {
            throw new Error(`Cannot connect to node of type ${prev.constructor.name}`);
        }

        if (!GraphNode.shapeMatch(this._inShape, prevOutShape)) {
            throw new Error(`Shape mismatch: Cannot connect Tensor with input shape [${this._inShape}] to previous node with output shape [${prevOutShape}]`);
        }

        this._prev = prev;

        if (prev instanceof BranchOp && indexPrev !== undefined) {
            (prev as BranchOp)._nexts[indexPrev] = this;
        } else {
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("Tensor already has a sink connection");
        }

        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof MergeOp) {
            if (indexNext === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            const validatedNextIndex = GraphNode.validateIndex(indexNext, next.inShape.length, "Tensor.connectSink (MergeOp input)");
            nextInShape = next.inShape[validatedNextIndex] as number[];
            indexNext = validatedNextIndex;
        } else if (next instanceof BranchOp) {
            nextInShape = next.inShape as number[];
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }

        if (!GraphNode.shapeMatch(this._outShape, nextInShape)) {
            throw new Error(`Shape mismatch: Cannot connect Tensor with output shape [${this._outShape}] to next node with input shape [${nextInShape}]`);
        }

        this._next = next;

        if (next instanceof MergeOp && indexNext !== undefined) {
            (next as MergeOp)._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(indexSelf?: number): void {
        if (this._prev) {
            if (this._prev instanceof BranchOp) {
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
            if (this._next instanceof MergeOp) {
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

    to_torch_functional(inputs: string[]): string {
        return `${inputs[0]} = ${inputs[0]}`;
    }
} 