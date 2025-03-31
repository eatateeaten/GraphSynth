import { GraphNode } from './graph_node';

export abstract class MergeOp extends GraphNode {
    protected _inShapes: number[][];
    protected _outShape: number[];
    public _prevs: GraphNode[] = [];
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;
    

    constructor(
        id: string,
        inShapes: number[][],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, target);
        this._inShapes = inShapes;
        this._opType = opType;
        this._params = params;
        this._outShape = this.computeOutShape();
    }

    protected abstract computeOutShape(): number[];
    abstract to_torch_functional(inputs: string[], outputs?: string[]): string;

    // Getters and setters
    get inShape(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params as Record<string, any>) = { ...params };
        
        // Recalculate output shape
        try {
            this._outShape = this.computeOutShape();
        } catch (err: any) {
            // If shape inference fails, we keep the existing output shape
            console.warn(`Failed to update output shape after params change: ${err.message}`);
        }
    }

    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            throw new Error("MergeOp.addPrev requires an input index");
        }
        
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._inShapes.length, "MergeOp.addPrev");
        
        if (this._prevs[validatedIndex] !== null && this._prevs[validatedIndex] !== undefined) {
            throw new Error(`MergeOp already has a connection at input ${validatedIndex}`);
        }
        
        this._prevs[validatedIndex] = prev;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("MergeOp already has a sink connection");
        }
        
        // Just set our next reference - Graph handles all validation and connections
        this._next = next;
    }

    deletePrev(indexSelf?: number): void {
        if (indexSelf === undefined) {
            this._prevs.fill(null as unknown as GraphNode);
            return;
        }
        
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._inShapes.length, "MergeOp.deletePrev");
        
        this._prevs[validatedIndex] = null as unknown as GraphNode;
    }

    deleteNext(indexSelf?: number): void {
        // Just clear our next reference
        this._next = null;
    }

    get prev(): GraphNode | null {
        return this._prevs.length > 0 ? this._prevs[0] : null;
    }

    set prev(node: GraphNode | null) {
        this._prevs = [];
        if (node !== null) {
            this._prevs.push(node);
        }
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
        super(id, [], target, opType, params);
    }

    protected computeOutShape(): number[] {
        if (this._prevs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }
        if (!this._prevs[0] || !this._prevs[1]) {
            throw new Error("PointwiseOp requires both inputs");
        }
        
        const shape0 = this._prevs[0].outShape;
        const shape1 = this._prevs[1].outShape;
        
        if (!shape0 || !shape1) {
            throw new Error("PointwiseOp requires both inputs to have defined shapes");
        }
        
        // Convert to number[] if needed
        const shape0Array = (Array.isArray(shape0) ? shape0 : [shape0]) as number[];
        const shape1Array = (Array.isArray(shape1) ? shape1 : [shape1]) as number[];
        
        if (!GraphNode.shapeMatch(shape0Array, shape1Array)) {
            throw new Error("PointwiseOp requires input shapes to match");
        }
        
        return [...shape0Array];
    }

    addPrev(prev: GraphNode, indexSelf: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            throw new Error("PointwiseOp.addPrev requires an input index");
        }
        
        if (indexSelf < 0 || indexSelf >= 2) {
            throw new Error(`PointwiseOp can only have 2 inputs. Invalid index: ${indexSelf}`);
        }
        
        if (this._prevs[indexSelf] !== null && this._prevs[indexSelf] !== undefined) {
            throw new Error(`PointwiseOp already has a connection at input ${indexSelf}, disconnect first`);
        }

        // Initialize _prevs array if needed
        if (this._prevs.length < 2) {
            this._prevs = new Array(2).fill(null);
        }

        this._prevs[indexSelf] = prev;
    }
}
