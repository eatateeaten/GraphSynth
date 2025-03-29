import { GraphNode } from './graph';
import { Tensor } from './tensor';
import { Op } from './op';
import { MergeOp } from './merge_op';

export abstract class BranchOp extends GraphNode {
    protected _inShape: number[];
    protected _outShapes: number[][];
    protected _prev: GraphNode | null = null;
    public _nexts: GraphNode[] = [];
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;

    constructor(
        id: string,
        inShape: number[],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, target);
        this._inShape = inShape;
        this._opType = opType;
        this._params = params;
        this._outShapes = this.computeOutShapes();
    }

    protected abstract computeOutShapes(): number[][];
    abstract to_torch_functional(inputs: string[]): string;

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[][] { return this._outShapes; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return null; }
    set next(node: GraphNode | null) { /* Do nothing */ }
    get nexts(): GraphNode[] { return this._nexts; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("BranchOp already has a source connection");
        }

        let prevOutShape: number[];
        if (prev instanceof Tensor || prev instanceof Op) {
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            if (indexPrev !== undefined) {
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "BranchOp.connectSource (BranchOp output)");
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
            throw new Error(`Shape mismatch: Cannot connect BranchOp with input shape [${this._inShape}] to previous node with output shape [${prevOutShape}]`);
        }

        this._prev = prev;

        if (prev instanceof BranchOp && indexPrev !== undefined) {
            prev._nexts[indexPrev] = this;
        } else {
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            throw new Error("BranchOp requires an output index for connection");
        }
        
        const validatedIndex = GraphNode.validateIndex(indexSelf, this._outShapes.length, "BranchOp.connectSink");

        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof MergeOp) {
            if (indexNext === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            const validatedNextIndex = GraphNode.validateIndex(indexNext, next.inShape.length, "BranchOp.connectSink (MergeOp input)");
            nextInShape = next.inShape[validatedNextIndex] as number[];
            indexNext = validatedNextIndex;
        } else if (next instanceof BranchOp) {
            nextInShape = next.inShape;
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }

        if (!GraphNode.shapeMatch(this._outShapes[validatedIndex], nextInShape)) {
            throw new Error(`Shape mismatch at index ${validatedIndex}: Cannot connect BranchOp with output shape [${this._outShapes[validatedIndex]}] to next node with input shape [${nextInShape}]`);
        }

        this._nexts[validatedIndex] = next;

        if (next instanceof MergeOp && indexNext !== undefined) {
            (next as MergeOp)._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(indexSelf?: number): void {
        if (this._prev) {
            if (this._prev instanceof BranchOp) {
                const branchIndex = this._prev._nexts.indexOf(this);
                if (branchIndex >= 0) {
                    this._prev._nexts[branchIndex] = null as unknown as GraphNode;
                }
            } else {
                this._prev.next = null;
            }
            this._prev = null;
        }
    }

    disconnectSink(indexSelf?: number): void {
        if (indexSelf !== undefined) {
            const validatedIndex = GraphNode.validateIndex(indexSelf, this._outShapes.length, "BranchOp.disconnectSink");
            this._disconnectSinkAtIndex(validatedIndex);
        } else {
            for (let i = 0; i < this._nexts.length; i++) {
                if (this._nexts[i]) {
                    this._disconnectSinkAtIndex(i);
                }
            }
        }
    }

    private _disconnectSinkAtIndex(index: number): void {
        const next = this._nexts[index];
        if (next) {
            if (next instanceof MergeOp) {
                const mergeIndex = (next as MergeOp)._prevs.indexOf(this);
                if (mergeIndex >= 0) {
                    (next as MergeOp)._prevs[mergeIndex] = null as unknown as GraphNode;
                }
            } else {
                next.prev = null;
            }
            this._nexts[index] = null as unknown as GraphNode;
        }
    }
}

export class MapOp extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, inShape, target, opType, params);
    }

    protected computeOutShapes(): number[][] {
        const numOutputs = this._params.numOutputs || 1;
        const outShapes: number[][] = [];
        
        for (let i = 0; i < numOutputs; i++) {
            outShapes.push([...this._inShape]);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[]): string {
        const op = this._opType.toLowerCase();
        const numOutputs = this._params.numOutputs || 1;
        
        if (numOutputs === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        const outputs = inputs.map((input, i) => `${input}_${i}`);
        const assignments = outputs.map((output, i) => `${output} = ${inputs[0]}`).join('\n');
        
        return assignments;
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                indexSelf = this._nexts.length;
                if (indexSelf >= this._outShapes.length) {
                    this._outShapes.push([...this._inShape]);
                }
            }
        }
        
        super.connectSink(next, indexSelf, indexNext);
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }
}

export class Broadcast extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        params: { numOutputs: number }
    ) {
        super(id, inShape, target, "Broadcast", params);
    }

    protected computeOutShapes(): number[][] {
        const numOutputs = this._params.numOutputs;
        const outShapes: number[][] = [];
        
        for (let i = 0; i < numOutputs; i++) {
            outShapes.push([...this._inShape]);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[]): string {
        const numOutputs = this._params.numOutputs;
        
        if (numOutputs === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        const outputs = inputs.map((input, i) => `${input}_${i}`);
        const assignments = outputs.map((output, i) => `${output} = ${inputs[0]}`).join('\n');
        
        return assignments;
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                indexSelf = this._nexts.length;
                if (indexSelf >= this._outShapes.length) {
                    this._outShapes.push([...this._inShape]);
                }
            }
        }
        
        super.connectSink(next, indexSelf, indexNext);
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }
}

export class Split extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        params: { dim: number, sections: number[] }
    ) {
        super(id, inShape, target, "Split", params);
    }

    protected computeOutShapes(): number[][] {
        const { dim, sections } = this._params;
        const outShapes: number[][] = [];
        
        for (const size of sections) {
            const outShape = [...this._inShape];
            outShape[dim] = size;
            outShapes.push(outShape);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[]): string {
        const { dim, sections } = this._params;
        
        if (sections.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        const outputsStr = inputs.map((input, i) => `${input}`).join(', ');
        return `${outputsStr} = torch.split(${inputs[0]}, sections=${JSON.stringify(sections)}, dim=${dim})`;
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }

    // For test compatibility
    get nexts(): GraphNode[] {
        return this._nexts;
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                indexSelf = this._nexts.length;
                if (indexSelf >= this._outShapes.length) {
                    // This is not ideal for Split since output shapes depend on sections
                    // But we'll allow it for flexibility
                    const lastShape = [...this._inShape];
                    lastShape[this._params.dim] = 1; // Default to size 1 for new sections
                    this._outShapes.push(lastShape);
                    // Update sections parameter
                    this._params.sections.push(1);
                }
            }
        }
        
        super.connectSink(next, indexSelf, indexNext);
    }
} 