/**
 * Base types for the graph implementation.
 * This file is used to prevent circular imports.
 */

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

    // Abstract shape and parameter accessors
    abstract get inShape(): number[] | null;
    abstract get outShape(): number[] | null;
    abstract get params(): Record<string, any>;

    abstract addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void;
    abstract addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract deletePrev(indexSelf?: number): void;
    abstract deleteNext(indexSelf?: number): void;
    abstract to_torch_functional(inputs: string[]): string;

    static checkIndexInBound(index: number, length: number, context: string): number {
        if (index < 0 || index >= length) {throw new Error(`${context}: Index ${index} is out of bounds for length ${length}`);}
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