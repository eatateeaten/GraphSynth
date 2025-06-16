import { GraphNode } from './graph_node';
import { Tensor } from './tensor';

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
    private _inputs: Tensor[] = [];
    private _outputs: Tensor[] = [];
    protected readonly _target: string;

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, params);
        this._inShape = [];
        this._outShape = [];
        this._opType = opType;
        this._params = params;
        this._target = target;
    }

    // Shape getters
    get inShape(): number[][] { return this._inShape; }
    get outShape(): number[][] { return this._outShape; }
    get inShapes(): number[][] { return this._inShape; }
    get outShapes(): number[][] { return this._outShape; }

    // Tensor getters
    get inputs(): Tensor[] { return this._inputs; }
    get outputs(): Tensor[] { return this._outputs; }

    // Operation and parameter getters/setters
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) { this._params = { ...params }; }

    /**
     * Abstract method to compute output shapes based on input shapes
     */
    protected abstract computeOutShape(): number[][];

    /**
     * Abstract method to generate framework-specific code
     */
    abstract emitTorchFunctional(inputs: string[], outputs?: string[]): string;
    abstract emitIR(): string;

    /**
     * Generates framework-specific code
     */
    emitTorch(inputs: string[], outputs?: string[]): string {
        if (this._target !== "Torch") {
            throw new Error(`Code generation not implemented for target framework: ${this._target}`);
        }
        return this.emitTorchFunctional(inputs, outputs);
    }

    /**
     * Adds a previous node to this module (implements GraphNode.addPrev) 
     */
    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
        if (!prev.outShapes) {
            throw new Error("Input node must have a defined shape");
        }

        // Process the outShape from the node
        let nodeShape: number[][];
        if (Array.isArray(prev.outShapes[0])) {
            // It's already a number[][]
            nodeShape = prev.outShapes as number[][];
        } else {
            // It's a number[], wrap it in an array
            nodeShape = [prev.outShapes as unknown as number[]];
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
    addNext(next: GraphNode, indexSelf?: number): void {
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
            /* sophia: fix this. it should get the prevShape from the node */
            this.addPrev(node, []);
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