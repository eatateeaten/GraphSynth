import { GraphNode } from './graph_node';
import { forwardShapeInference, getTorchCode } from './torch_nn_module_op';
import { g_GraphConfig } from './config';

export class Op extends GraphNode {
    protected readonly _opType: string;

    constructor(
        id: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, params);
        this._inShapes = [null];
        this._outShapes = [null];
        this._prevs = [null];
        this._nexts = [null];
        this._opType = opType;
    }

    protected computeOutShape(): number[] {
        if (this._inShapes[0] === null) {
            throw new Error(`Cannot compute output shape without input shape for operation ${this._opType}`);
        }

        // Use forwardShapeInference for torch operations
        if (g_GraphConfig.target === "Torch") {
            try {
                return forwardShapeInference(this._opType, this._inShapes[0], this._params);
            } catch (err: any) {
                throw new Error(`Shape inference error for ${this._opType}: ${err.message}. Consider using a different set of parameters.`);
            }
        }

        // For non-torch operations, throw an error
        throw new Error(`No shape inference implementation available for target '${ g_GraphConfig.target }' and operation '${this._opType}'`);
    }

    to_torch_functional(inputs: string[], outputs: string[]): string {
        if (g_GraphConfig.target  !== "Torch") {
            throw new Error("Operation is not a PyTorch operation");
        }

        if (this._inShapes[0] === null || this._outShapes[0] === null) {
            throw new Error("Cannot generate torch code: operation has undefined input or output shape");
        }

        // No fallback - either get the module code or error out
        const moduleCode = getTorchCode(this._opType, this._params);
        return `${outputs[0]} = ${moduleCode}(${inputs[0]})`;
    }

    /**
     * Generates PyTorch code for this operation without requiring input variable names.
     * 
     * @returns A string containing the PyTorch code for this operation
     * @throws Error if the operation is not a PyTorch operation
     */
    to_torch(): string {
        if (g_GraphConfig.target !== "Torch") {
            throw new Error("Operation is not a PyTorch operation");
        }

        // No fallback - either get the module code or error out
        return getTorchCode(this._opType, this._params);
    }

    // Getters and setters
    set inShape(shape: number[] | null) { 
        // inShape can only be set during connection
        throw new Error("Cannot directly set inShape for Op. Connect a source node instead.");
    }
    get opType(): string { return this._opType; }
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

        // Set inShape and compute outShape
        this._inShapes = [[...prevOutShape]];

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
