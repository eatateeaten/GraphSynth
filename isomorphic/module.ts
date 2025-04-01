import { GraphNode } from './graph_node';

/**
 * Module represents a neural network module that can have multiple inputs and outputs.
 * It inherits from GraphNode and provides basic module functionality.
 */
export abstract class Module extends GraphNode {
    protected _inShape: number[][];
    protected _outShape: number[][];
    protected _prevs: GraphNode[] = [];
    protected _nexts: GraphNode[] = [];
    protected readonly _opType: string;
    protected _params: Record<string, any>;

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target);
        this._inShape = [];
        this._outShape = [];
        this._opType = opType;
        this._params = params;
    }

    /**
     * Returns the module's input shapes (required by GraphNode)
     */
    get inShape(): number[][] {
        return this._inShape;
    }

    /**
     * Returns the module's output shapes (required by GraphNode)
     */
    get outShape(): number[][] {
        return this._outShape;
    }

    /**
     * Returns the module's input shapes
     */
    get inShapes(): number[][] {
        return this._inShape;
    }

    /**
     * Returns the module's output shapes
     */
    get outShapes(): number[][] {
        return this._outShape;
    }

    /**
     * Returns the module's input nodes
     */
    get inNodes(): GraphNode[] {
        return this._prevs;
    }

    /**
     * Returns the module's output nodes
     */
    get outNodes(): GraphNode[] {
        return this._nexts;
    }

    /**
     * Returns the module's operation type
     */
    get opType(): string {
        return this._opType;
    }

    /**
     * Returns the module's parameters
     */
    get params(): Record<string, any> {
        return { ...this._params };
    }

    /**
     * Sets the module's parameters
     */
    set params(params: Record<string, any>) {
        this._params = { ...params };
    }

    /**
     * Abstract method to compute output shapes based on input shapes
     */
    protected abstract computeOutShape(): number[][];

    /**
     * Abstract method to generate framework-specific code
     */
    abstract to_torch_functional(inputs: string[], outputs?: string[]): string;

    /**
     * Adds a previous node to this module (implements GraphNode.addPrev)
     */
    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (!prev.outShape) {
            throw new Error("Input node must have a defined shape");
        }

        // Process the outShape from the node
        let nodeShape: number[][];
        if (Array.isArray(prev.outShape[0])) {
            // It's already a number[][]
            nodeShape = prev.outShape as number[][];
        } else {
            // It's a number[], wrap it in an array
            nodeShape = [prev.outShape as number[]];
        }

        if (indexSelf === undefined) {
            this._prevs.push(prev);
            this._inShape = [...this._inShape, ...nodeShape];
        } else {
            if (indexSelf < 0 || indexSelf > this._prevs.length) {
                throw new Error(`Invalid input index: ${indexSelf}`);
            }
            this._prevs.splice(indexSelf, 0, prev);
            
            // Insert the new shape(s) at the specified index
            this._inShape = [
                ...this._inShape.slice(0, indexSelf),
                ...nodeShape,
                ...this._inShape.slice(indexSelf)
            ];
        }
        this._outShape = this.computeOutShape();
    }

    /**
     * Adds a next node to this module (implements GraphNode.addNext)
     */
    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            this._nexts.push(next);
        } else {
            if (indexSelf < 0 || indexSelf > this._nexts.length) {
                throw new Error(`Invalid output index: ${indexSelf}`);
            }
            this._nexts.splice(indexSelf, 0, next);
        }
    }

    /**
     * Deletes a previous node from this module (implements GraphNode.deletePrev)
     */
    deletePrev(indexSelf?: number): void {
        if (indexSelf === undefined) {
            this._prevs = [];
            this._inShape = [];
            this._outShape = this.computeOutShape();
        } else {
            if (indexSelf < 0 || indexSelf >= this._prevs.length) {
                throw new Error(`Invalid input index: ${indexSelf}`);
            }
            this._prevs.splice(indexSelf, 1);
            this._inShape.splice(indexSelf, 1);
            this._outShape = this.computeOutShape();
        }
    }

    /**
     * Deletes a next node from this module (implements GraphNode.deleteNext)
     */
    deleteNext(indexSelf?: number): void {
        if (indexSelf === undefined) {
            this._nexts = [];
        } else {
            if (indexSelf < 0 || indexSelf >= this._nexts.length) {
                throw new Error(`Invalid output index: ${indexSelf}`);
            }
            this._nexts.splice(indexSelf, 1);
        }
    }

    /**
     * For GraphNode compatibility
     */
    get prev(): GraphNode | null {
        return this._prevs.length > 0 ? this._prevs[0] : null;
    }

    /**
     * For GraphNode compatibility
     */
    set prev(node: GraphNode | null) {
        // Clear existing nodes and add the new one
        this._prevs = [];
        this._inShape = [];
        if (node !== null) {
            this.addPrev(node);
        }
    }

    /**
     * For GraphNode compatibility
     */
    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    /**
     * For GraphNode compatibility
     */
    set next(node: GraphNode | null) {
        // Clear existing nodes and add the new one
        this._nexts = [];
        if (node !== null) {
            this.addNext(node);
        }
    }

    /**
     * Returns the module's state dict (parameters)
     */
    getStateDict(): Record<string, any> {
        return {
            params: this._params
        };
    }

    /**
     * Loads state dict into the module
     */
    loadStateDict(stateDict: Record<string, any>): void {
        this._params = { ...stateDict.params };
    }
} 