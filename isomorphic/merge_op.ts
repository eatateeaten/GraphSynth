import { GraphNode } from './graph_node';
import { getDifferentiablePointWiseOpCode } from './pointwise_op_map';


export abstract class MergeOp extends GraphNode {
    protected _inShape: number[][];
    protected _outShape: number[]| null;
    public _prevs: GraphNode[] = [];
    protected _next: GraphNode | null = null;
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;
    public _numberOfMerges: number; 

    constructor(
        id: string,
        target: string,
        opType: string,
        params: Record<string, any> = {}, 
        numberOfMerges: number 
    ) {
        super(id, target);
        this._inShape = Array(numberOfMerges).fill(null)
        this._opType = opType;
        this._params = params;
        this._outShape = null; 
        this._numberOfMerges = numberOfMerges
    }
    
    protected abstract computeOutShape(): number[];
    protected abstract checkIncomingShapeMatch(shape: number[]): void; 
    abstract to_torch_functional(inputs: string[], outputs?: string[]): string;
    

    // Getters and setters
    get inShape(): number[][] { return this._inShape; }
    get outShape(): number[] | null { return this._outShape; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params as Record<string, any>) = { ...params };
        // Recalculate output shape
        this._outShape = this.computeOutShape();
    }

    // addPrev 
    // Super(addPrev) 
    // main class addPrev should only take care of checkingIncomingShapeValidity. And this can be done for most Merge Operations 
    // For Reduceable Op, at any this stage they can compute an outShape (Reduceable Op's computeOutShape can be just an operation over the existing outShape)
    // However, For non-reduceable Op, they will have to check that they have filled all the requireed inputs before they can compute and outShape 

    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            throw new Error("MergeOp.addPrev requires an input index"); // a bit redundant if calling this from Graph.ts's connect 
        } 
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._inShape.length, "MergeOp.addPrev"); // a bit redundant if calling this from Graph.ts's connect 
        if (this._prevs[validatedIndex] !== null && this._prevs[validatedIndex] !== undefined) {
            throw new Error(`MergeOp already has a connection at input ${validatedIndex}`); // a bit redundant 
        }
        //-------------------------------------------------------
        this.checkIncomingShapeMatch(prevOutShape); 
        if( this._numberOfMerges === this._prevs.filter(x => x != null).length + 1) {
            this.computeOutShape(); 
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
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._inShape.length, "MergeOp.deletePrev");

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
        params: Record<string, any> = {},
        inputSize: number = 2
    ) {
        super(id, target, opType, params, inputSize);
    }

    protected checkIncomingShapeMatch(shape: number[]): number[] {
        /* sophia: implement this */
        return [];
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

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length !== 2) {
            throw new Error("PointwiseOp requires exactly 2 inputs");
        }

        const opCode = getDifferentiablePointWiseOpCode(this._opType, this._target);
        return `${inputs[0]} = ${opCode}(${inputs[0]}, ${inputs[1]})`;
    }

    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
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
