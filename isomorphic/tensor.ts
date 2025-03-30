import { GraphNode } from './graph_node';

export class Tensor extends GraphNode {
    protected _inShape: number[];
    protected _outShape: number[];
    protected _prev: GraphNode | null = null;
    protected _next: GraphNode | null = null;
    protected _variableName: string | null = null;

    constructor(id: string, shape: number[], target: string, variableName: string | null = null) {
        super(id, target);
        this._inShape = shape;
        this._outShape = shape;
        this._variableName = variableName;
    }

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[] { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get variableName(): string | null { return this._variableName; }
    set variableName(name: string | null) { this._variableName = name; }
    
    // Implement params accessors
    get params(): Record<string, any> { 
        return { 
            shape: [...this._inShape],
            variableName: this._variableName
        }; 
    }
    set params(params: Record<string, any>) {
        if (params.shape) {
            this._inShape = [...params.shape];
            this._outShape = [...params.shape];
        }
        if ('variableName' in params) {
            this._variableName = params.variableName;
        }
    }

    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("Tensor already has a source connection");
        }
        // Just set our prev reference - Graph handles all validation and connections
        this._prev = prev;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("Tensor already has a sink connection");
        }
        // Just set our next reference - Graph handles all validation and connections
        this._next = next;
    }

    deletePrev(indexSelf?: number): void {
        // Just clear our reference
        this._prev = null;
    }

    deleteNext(indexSelf?: number): void {
        // Just clear our reference
        this._next = null;
    }

    to_torch_functional(inputs: string[]): string {
        if (inputs.length === 0) {
            // This is a source tensor - use variableName if available, otherwise ID
            return this._variableName || this.id;
        }
        return `${inputs[0]} = ${inputs[0]}`;
    }
} 