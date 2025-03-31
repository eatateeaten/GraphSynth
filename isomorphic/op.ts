import { GraphNode } from './graph_node';
import { getElementwiseOpCode, forwardShapeInference, getTorchCode } from './torch_nn_module_op';
import { Tensor } from './tensor';
import { BranchOp } from './branch_op';
import { MergeOp } from './merge_op';

export class Op extends GraphNode {
    protected _inShape: number[] | null = null;
    protected _outShape: number[] | null = null;
    protected _prev: GraphNode | null = null;
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, target);
        this._opType = opType;
        this._params = params;
    }

    protected computeOutShape(): number[] {
        if (!this._inShape) {
            throw new Error(`Cannot compute output shape without input shape for operation ${this._opType}`);
        }
        
        // Use forwardShapeInference for torch operations
        if (this._target === "torch") {
            try {
                return forwardShapeInference(this._opType, this._inShape, this._params);
            } catch (err: any) {
                throw new Error(`Shape inference error for ${this._opType}: ${err.message}. Consider using a different set of parameters.`);
            }
        }
        
        // For non-torch operations, throw an error
        throw new Error(`No shape inference implementation available for target '${this._target}' and operation '${this._opType}'`);
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        
        if (this._inShape === null || this._outShape === null) {
            throw new Error("Cannot generate torch code: operation has undefined input or output shape");
        }
        
        // No fallback - either get the module code or error out
        const moduleCode = getTorchCode(this._opType, this._params);
        return `${inputs[0]} = ${moduleCode}(${inputs[0]})`;
    }

    /**
     * Generates PyTorch code for this operation without requiring input variable names.
     * 
     * @returns A string containing the PyTorch code for this operation
     * @throws Error if the operation is not a PyTorch operation
     */
    to_torch(): string {
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        
        // No fallback - either get the module code or error out
        return getTorchCode(this._opType, this._params);
    }

    // Getters and setters
    get inShape(): number[] | null { return this._inShape; }
    set inShape(shape: number[] | null) { 
        // inShape can only be set during connection
        throw new Error("Cannot directly set inShape for Op. Connect a source node instead.");
    }
    get outShape(): number[] | null { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // We need to make a deep copy to avoid modifying the original object
        (this._params as Record<string, any>) = { ...params };
        
        // Recalculate output shape if input shape is available
        if (this._inShape) {
            try {
                this._outShape = this.computeOutShape();
            } catch (err: any) {
                // If shape inference fails, we keep the existing output shape
                console.warn(`Failed to update output shape after params change: ${err.message}`);
            }
        }
    }

    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("Op already has a source connection");
        }

        // Get the output shape from the source node
        let sourceOutShape: number[];
        
        // Extract shape from prev node based on its type
        if (prev instanceof BranchOp && indexPrev !== undefined) {
            const branchOutShape = prev.outShape[indexPrev];
            if (branchOutShape === null || branchOutShape === undefined) {
                throw new Error(`Cannot connect to BranchOp with id ${prev.id} at output ${indexPrev}: output shape is undefined`);
            }
            sourceOutShape = branchOutShape;
        } else if (prev instanceof Tensor) {
            if (!prev.outShape) {
                throw new Error(`Cannot connect to Tensor with id ${prev.id}: output shape is undefined`);
            }
            sourceOutShape = prev.outShape;
        } else if (prev instanceof Op) {
            if (!prev.outShape) {
                throw new Error(`Cannot connect to Op with id ${prev.id}: output shape is undefined`);
            }
            sourceOutShape = prev.outShape;
        } else if (prev instanceof MergeOp) {
            if (!prev.outShape) {
                throw new Error(`Cannot connect to MergeOp with id ${prev.id}: output shape is undefined`);
            }
            sourceOutShape = prev.outShape;
        } else {
            throw new Error(`Cannot connect to unknown node type: ${prev.constructor.name}`);
        }
        
        // Set inShape and compute outShape
        this._inShape = [...sourceOutShape];
        
        try {
            this._outShape = this.computeOutShape();
        } catch (err: any) {
            // Reset inShape if shape inference fails
            this._inShape = null;
            throw err;
        }
        
        // Set our prev reference
        this._prev = prev;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._next !== null) {
            throw new Error("Op already has a sink connection");
        }
        this._next = next;
    }

    deletePrev(indexSelf?: number): void {
        if (this._prev) {
            // Just clear our reference and reset shapes
            this._prev = null;
            this._inShape = null;
            this._outShape = null;
        }
    }

    deleteNext(indexSelf?: number): void {
        // Just clear our next reference
        this._next = null;
    }
} 