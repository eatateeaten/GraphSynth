import { GraphNode } from './graph_node';
import { ParamError, ShapeMatchError } from './errors';
import { ModuleDB } from '../moduledb';

export abstract class MergeOp extends GraphNode {
    protected readonly _opType: string;
    protected _numberOfMerges: number; 

    constructor(
        id: string,
        opType: string,
        numberOfMerges: number,
        params: Record<string, any> = {}, 
    ) {
        super(id, params);
        this._inShapes = Array(numberOfMerges).fill(null);
        this._outShapes = [null];
        this._prevs = Array(numberOfMerges).fill(null); 
        this._nexts = [null];
        this._opType = opType;
        this._numberOfMerges = numberOfMerges; 
    }

    /* this is probably required for all nodes??? why is it defined here. */
    protected abstract computeOutShape(): number[] | null;
    protected abstract checkIncomingShapeMatch(shape: number[]): void; 
    abstract emitTorchModule(inputs: string[], outputs?: string[]): string;
    abstract emitIR(): string;

    // Getters and setters
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params) = { ...params };
        // Recalculate output shape
        this._outShapes = [this.computeOutShape()];
    }

    // addPrev 
    // Super(addPrev) 
    // main class addPrev should only take care of checkingIncomingShapeValidity. And this can be done for most Merge Operations 
    // For Reduceable Op, at any this stage they can compute an outShape (Reduceable Op's computeOutShape can be just an operation over the existing outShape)
    // However, For non-reduceable Op, they will have to check that they have filled all the requireed inputs before they can compute and outShape 
    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number): void {
        if (indexSelf === undefined) {
            throw new Error("MergeOp.addPrev requires an input index"); // a bit redundant if calling this from Graph.ts's connect 
        } 

        /* Different for every derived object */
        this.checkIncomingShapeMatch(prevOutShape);

        // Store both the prev node and its shape
        this._prevs[indexSelf] = prev;
        this._inShapes[indexSelf] = [...prevOutShape];
        
        // Now compute the output shape based on the updated input shapes
        this._outShapes = [this.computeOutShape()];
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (this._nexts[0] !== null) {
            throw new Error("MergeOp already has a sink connection");
        }

        // Just set our next reference - Graph handles all validation and connections
        this._nexts[0] = next;
    }

    deletePrev(indexSelf: number): void {
        //at this point we have already check that indexSelf is valid from the function calling deletePrev
        this._prevs[indexSelf] = null;
        // Just clear our reference and reset shapes 
        this._inShapes[indexSelf] = null;
        this._outShapes = [null]; 
    }

    deleteNext(): void {
        // Just clear our next reference
        this._nexts[0] = null;
    }
}


/**
 * PointwiseOp represents operations that take exactly two inputs with matching shapes
 * and perform element-wise operations between them.
 */
export class PointwiseOp extends MergeOp {
    constructor(
        id: string,
        opType: string,
        params: Record<string, any> = {}
    ) {
        super(id, opType, 2, params); // Always 2 inputs for pointwise ops
    }

    static fromParams(id: string, params: Record<string, any>): PointwiseOp {
        if (!params.opType)
            throw new ParamError("Operation type is required for PointwiseOp");
        return new PointwiseOp(id, params.opType, params);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (!this._inShapes.some(s => s !== null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) return;

        if (shape.length !== referenceShape.length) {
            throw new ShapeMatchError(`Shape rank mismatch: expected ${referenceShape.length}, got ${shape.length}`);
        }

        for (let i = 0; i < shape.length; i++) {
            if (shape[i] !== referenceShape[i]) {
                throw new ShapeMatchError(`Shape mismatch at dim ${i}: expected ${referenceShape[i]}, got ${shape[i]}`);
            }
        }
    }

    /* TODO: these checks are good. need a canonical place to put these for each concrete class */
    protected computeOutShape(): number[] | null {
        if (this._prevs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }

        const shape = this._inShapes[0];
        if (shape === null) {
            throw new Error("PointwiseOp requires first input to have defined shape");
        }

        return shape;
    }

    emitTorchModule(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }
        const moduleDef = ModuleDB.get('PointwiseOp');
        if (!moduleDef) {
            throw new Error("PointwiseOp not found in ModuleDB");
        }
        const torchOp = moduleDef.toPytorchModule(this._params);
        return `${inputs[0]} = ${torchOp}(${inputs[0]}, ${inputs[1]})`;
    }

    emitIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `${this._opType}(${JSON.stringify(this._params)}) -> ${shapeStr}`;
    }
}

/**
 * DotOp represents dot product operations between two tensors.
 * For 1D tensors: dot product
 * For 2D tensors: matrix multiplication
 * For higher dimensions: batched matrix multiplication
 */
export class DotOp extends MergeOp {
    constructor(
        id: string,
        params: Record<string, any> = {}
    ) {
        super(id, "Dot", 2, params); // Always 2 inputs for dot ops
    }

    static fromParams(id: string, params: Record<string, any>): DotOp {
        return new DotOp(id, params);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (this._inShapes.every(s => s === null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) return;

        // For dot product, last dimension of first tensor must match first dimension of second tensor
        if (referenceShape[referenceShape.length - 1] !== shape[0]) {
            throw new Error(
                `Dot product dimension mismatch: last dim of first tensor (${referenceShape[referenceShape.length - 1]}) ` +
                `must match first dim of second tensor (${shape[0]})`
            );
        }
    }

    protected computeOutShape(): number[] | null {
        if (this._prevs.length !== 2) {
            // throw new Error("DotOp requires exactly 2 inputs");
            return null;
        }

        const shape1 = this._inShapes[0];
        const shape2 = this._inShapes[1];
        if (!shape1 || !shape2) {
            //throw new Error("DotOp requires both inputs to have defined shapes");
            return null;
        }

        // For dot product, output shape is [batch_dims..., shape1[-2], shape2[-1]]
        return [...shape1.slice(0, -1), shape2[shape2.length - 1]];
    }

    emitTorchModule(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("DotOp requires exactly 2 inputs");
        }
        const moduleDef = ModuleDB.get('DotOp');
        if (!moduleDef) {
            throw new Error("DotOp not found in ModuleDB");
        }
        const torchOp = moduleDef.toPytorchModule(this._params);
        return `${inputs[0]} = ${torchOp}(${inputs[0]}, ${inputs[1]})`;
    }

    emitIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `Dot(${JSON.stringify(this._params)}) -> ${shapeStr}`;
    }
}

/**
 * CrossOp represents cross product operations between two tensors.
 * Only valid for 3D vectors (shape [..., 3]).
 */
export class CrossOp extends MergeOp {
    constructor(
        id: string,
        params: Record<string, any> = {}
    ) {
        super(id, "Cross", 2, params); // Always 2 inputs for cross ops
    }

    static fromParams(id: string, params: Record<string, any>): CrossOp {
        return new CrossOp(id, params);
    }

    protected checkIncomingShapeMatch(shape: number[]): void {
        if (this._inShapes.every(s => s === null)) {
            return; // First shape, no need to check
        }

        const referenceShape = this._inShapes.find(s => s !== null);
        if (!referenceShape) return;

        // For cross product, last dimension must be 3
        if (shape[shape.length - 1] !== 3) {
            throw new Error(
                `Cross product requires 3D vectors, got shape [..., ${shape[shape.length - 1]}]`
            );
        }

        // All other dimensions must match for batched operations
        for (let i = 0; i < shape.length - 1; i++) {
            if (shape[i] !== referenceShape[i]) {
                throw new Error(
                    `Batch dimension mismatch at dim ${i}: ` +
                    `expected ${referenceShape[i]}, got ${shape[i]}`
                );
            }
        }
    }

    /* can be simplified once we find a canonical place for those checks */
    protected computeOutShape(): number[] | null {
        if (this._prevs.length !== 2) {
            //throw new Error("CrossOp requires exactly 2 inputs");
            return null;
        }

        const shape = this._inShapes[0];
        if (shape === null) {
            //throw new Error("CrossOp requires first input to have defined shape");
            return null;
        }

        // Cross product preserves the input shape
        return shape;
    }

    emitTorchModule(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("CrossOp requires exactly 2 inputs");
        }
        const moduleDef = ModuleDB.get('CrossOp');
        if (!moduleDef) {
            throw new Error("CrossOp not found in ModuleDB");
        }
        const torchOp = moduleDef.toPytorchModule(this._params);
        return `${inputs[0]} = ${torchOp}(${inputs[0]}, ${inputs[1]})`;
    }

    emitIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `Cross(${JSON.stringify(this._params)}) -> ${shapeStr}`;
    }
}
