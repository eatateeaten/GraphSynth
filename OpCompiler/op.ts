import { GraphNode } from './graph_node';
import { ShapeInferenceError, ParamError } from './errors';
import { ModuleDB, ModuleDef } from '../moduledb';

/**
 * Op represents operations with exactly one input and one output.
 * 
 * This class handles standard neural network operations like ReLU, Conv2d, 
 * Linear, etc. that take a single tensor input and produce a single tensor output.
 * For operations requiring multiple inputs or outputs, use MergeOp or BranchOp instead. */
export class Op extends GraphNode {
    protected readonly _module: ModuleDef;

    constructor(
        id: string,
        moduleName: string,
        params: Record<string, any>
    ) {
        super(id, params);
        this._inShapes = [null];
        this._outShapes = [null];
        this._prevs = [null];
        this._nexts = [null];
        this._module = ModuleDB.get(moduleName);
        this._shapeInferred = true;
    }


    protected computeOutShape(): number[] {
        // Op is guaranteed to have shape inference function
        const outputShapes = this._module.inferOutputShape!(this._inShapes[0]!, this._params);
        return outputShapes;
    }

    toTorchModule(): string {
        return this._module.emitPytorchModule(this._params);
    }

    /**
     * Generates intermediate representation for this operation.
     * 
     * @returns A string containing the IR representation of this operation
     */
    toIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `${this._module.label}(${JSON.stringify(this._params)}) -> ${shapeStr}`;
    }

    // Getters and setters
    set inShape(shape: number[] | null) { 
        // inShape can only be set during connection
        throw new Error("Cannot directly set inShape for Op. Connect a source node instead.");
    }
    get opType(): string { return this._module.label; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // We need to make a deep copy to avoid modifying the original object
        (this._params) = { ...params };
        
        // Recalculate output shape if input shape is available
        if (this._inShapes[0]) {
            try {
                this._outShapes[0] = this.computeOutShape();
            } catch (err: any) {
                // If shape inference fails, we keep the existing output shape
                console.warn(`Failed to update output shape after params change: ${err.message}`);
            }
        }
    }

    addPrev(prev: GraphNode, prevOutShape: number[]): void {
        if (this._prevs[0] !== null) {
            throw new Error("Op already has a source connection");
        }
        // Get the output shape from the source node        
        if (!prev.outShapes[0]) {
            throw new Error(`Previous node ${prev.id} has no output shape defined`);
        }

        // Set inShape and validate - Op is guaranteed to have validation function
        this._inShapes = [[...prevOutShape]];

        const errors = this._module.validateInputShape!(prevOutShape, this._params);
        if (errors.length > 0) {
            this._inShapes = [null];
            throw new ShapeInferenceError(`Shape validation failed: ${errors.join(', ')}`);
        }

        try {
            this._outShapes = [this.computeOutShape()];
        } catch (err: any) {
            // Reset inShape if shape inference fails
            this._inShapes = [null];
            throw err;
        }
        // Set our prev reference
        this._prevs[0] = prev;
    }

    addNext(next: GraphNode): void {
        if (this._nexts[0] !== null) {
            throw new Error("Op already has a sink connection");
        }
        this._nexts[0] = next;
    }

    deletePrev(): void {
        this._inShapes = [null];
        this._outShapes = [null];
        this._prevs = [null];
    }

    deleteNext(): void {
        this._nexts = [null];
    }
}
