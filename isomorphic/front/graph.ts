// torch.abs(input)       # Absolute value
// torch.sqrt(input)      # Square root
// torch.square(input)    # Square
// torch.exp(input)       # Exponential
// torch.log(input)       # Natural logarithm
// torch.sin(input)
// torch.cos(input)
// torch.tan(input)
// torch.asin(input)
// torch.acos(input)
// torch.atan(inpu

//The Graph  
//Merge Node 
//Shape check for Merge and Branch 

import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp } from './branch_op';
import { MergeOp, Concat } from './merge_op';
import { Split } from './branch_op';

export { Tensor, Op, Concat, Split, BranchOp, MergeOp };

export abstract class GraphNode {
    protected readonly _id: string;
    protected readonly _target: string;

    constructor(id: string, target: string) {
        // Validate UUID format
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        if (!uuidRegex.test(id)) {
            throw new Error(`Invalid UUID format: ${id}`);
        }
        this._id = id;
        this._target = target;
    }

    get id(): string { return this._id; }
    get target(): string { return this._target; }

    abstract get prev(): GraphNode | null;
    abstract set prev(node: GraphNode | null);
    abstract get next(): GraphNode | null;
    abstract set next(node: GraphNode | null);

    abstract connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void;
    abstract connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract disconnectSource(indexSelf?: number): void;
    abstract disconnectSink(indexSelf?: number): void;
    abstract to_torch_functional(inputs: string[]): string;

    static validateIndex(index: number, length: number, context: string): number {
        if (index < 0 || index >= length) {
            throw new Error(`${context}: Index ${index} is out of bounds for length ${length}`);
        }
        return index;
    }

    static shapeMatch(shape1: number[], shape2: number[]): boolean {
        if (shape1.length !== shape2.length) {
            return false;
        }
        for (let i = 0; i < shape1.length; i++) {
            if (shape1[i] !== shape2[i]) {
                return false;
            }
        }
        return true;
    }
}

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _sources: Set<GraphNode>;
    private _sinks: Set<GraphNode>;

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
    }

    addNode(node: GraphNode): void {
        if (this._nodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} already exists in graph`);
        }
        this._nodes.set(node.id, node);
        this._refreshNodeSinkSourceStatus(node);
    }

    removeNode(node: GraphNode): void {
        if (!this._nodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} does not exist in graph`);
        }

        // Disconnect all connections
        node.disconnectSource();
        node.disconnectSink();

        // Remove from collections
        this._nodes.delete(node.id);
        this._sources.delete(node);
        this._sinks.delete(node);
    }

    connect(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        if (!this._nodes.has(sink.id)) {
            throw new Error(`Sink node with id ${sink.id} does not exist in graph`);
        }

        sink.connectSource(source, sinkIndex, sourceIndex);
        source.connectSink(sink, sourceIndex, sinkIndex);

        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    disconnect(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        if (!this._nodes.has(sink.id)) {
            throw new Error(`Sink node with id ${sink.id} does not exist in graph`);
        }

        sink.disconnectSource(sinkIndex);
        source.disconnectSink(sourceIndex);

        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    getNode(id: string): GraphNode | undefined {
        return this._nodes.get(id);
    }

    getSources(): Set<GraphNode> {
        return new Set(this._sources);
    }

    getSinks(): Set<GraphNode> {
        return new Set(this._sinks);
    }

    private _refreshNodeSinkSourceStatus(node: GraphNode): void {
        // Check if node is a source (no incoming connections)
        if (node instanceof Tensor || node instanceof Op || node instanceof BranchOp) {
            if (node.prev === null) {
                this._sources.add(node);
            } else {
                this._sources.delete(node);
            }
        } else if (node instanceof MergeOp) {
            if (node._prevs.every(p => !p)) {
                this._sources.add(node);
            } else {
                this._sources.delete(node);
            }
        }

        // Check if node is a sink (no outgoing connections)
        if (node instanceof Tensor || node instanceof Op) {
            if (node.next === null) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        } else if (node instanceof BranchOp) {
            if (node._nexts.every(n => !n)) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        } else if (node instanceof MergeOp) {
            if (node.next === null) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        }
    }
}






