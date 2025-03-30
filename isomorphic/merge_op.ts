import { GraphNode } from './types';
import { getElementwiseOpCode } from './torch_nn_module_op';

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
    abstract to_torch_functional(inputs: string[]): string;

    // Getters and setters
    get inShape(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

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

export class Concat extends MergeOp {
    constructor(id: string, inShapes: number[][], target: string, params: { dim: number }) {
        super(id, inShapes, target, "Concat", params);
    }

    protected computeOutShape(): number[] {
        const dim = this._params.dim;
        if (dim < 0 || dim >= this._inShapes[0].length) {
            throw new Error(`Invalid concatenation dimension ${dim} for input shape of length ${this._inShapes[0].length}`);
        }

        const referenceShape = this._inShapes[0];
        for (let i = 1; i < this._inShapes.length; i++) {
            const shape = this._inShapes[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For concatenation, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (j !== dim && shape[j] !== referenceShape[j]) {
                    throw new Error(`For concatenation, input shapes must match on all dimensions except the concatenation dimension. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        const outShape = [...referenceShape];
        outShape[dim] = this._inShapes.reduce((sum, shape) => sum + shape[dim], 0);
        return outShape;
    }

    to_torch_functional(inputs: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }
}

export class ReduceOp extends MergeOp {
    constructor(
        id: string,
        inShapes: number[][],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, inShapes, target, opType, params);
    }

    protected computeOutShape(): number[] {
        if (this._inShapes.length < 1) {
            throw new Error("ReduceOp requires at least 1 input tensor");
        }

        const referenceShape = [...this._inShapes[0]];
        
        for (let i = 1; i < this._inShapes.length; i++) {
            const shape = this._inShapes[i];
            if (!GraphNode.shapeMatch(referenceShape, shape)) {
                throw new Error(`For reduction operations, all input shapes must match. Shape at index ${i} [${shape}] doesn't match reference shape [${referenceShape}]`);
            }
        }

        return referenceShape;
    }

    to_torch_functional(inputs: string[]): string {
        if (inputs.length < 1) {
            throw new Error("ReduceOp requires at least 1 input");
        }

        const op = this._opType.toLowerCase();
        
        if (inputs.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        let code = inputs[0];
        for (let i = 1; i < inputs.length; i++) {
            code = `torch.${op}(${code}, ${inputs[i]})`;
        }
        
        return `${inputs[0]} = ${code}`;
    }

    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._prevs.findIndex(p => !p);
            if (indexSelf === -1) {
                indexSelf = this._prevs.length;
                if (indexSelf >= this._inShapes.length) {
                    this._inShapes.push([...this._inShapes[0]]);
                }
            }
        }
        
        super.addPrev(prev, indexSelf, indexPrev);
    }
}

export class PointwiseReduce extends ReduceOp {
    constructor(id: string, inShapes: number[][], target: string, opType: string) {
        super(id, inShapes, target, opType, {});
    }

    protected computeOutShape(): number[] {
        if (this._inShapes.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 input tensors");
        }

        let resultShape = [...this._inShapes[0]];

        for (let i = 1; i < this._inShapes.length; i++) {
            const currentShape = this._inShapes[i];
            
            if (currentShape.length !== resultShape.length) {
                throw new Error(`Input shapes must have the same rank. Shape at index ${i} has rank ${currentShape.length}, expected ${resultShape.length}`);
            }
            
            resultShape = resultShape.map((dim, j) => {
                const otherDim = currentShape[j];
                
                if (dim === otherDim) {
                    return dim;
                }
                if (dim === 1) {
                    return otherDim;
                }
                if (otherDim === 1) {
                    return dim;
                }
                throw new Error(
                    `Incompatible shapes for broadcasting at dimension ${j}: ` +
                    `${dim} and ${otherDim}. Dimensions must be equal or one must be 1.`
                );
            });
        }

        return resultShape;
    }

    to_torch_functional(inputs: string[]): string {
        if (inputs.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 inputs");
        }
        const opCode = getElementwiseOpCode(this._opType);
        
        const result = inputs.reduce((acc, curr) => 
            acc ? `${opCode}(${acc}, ${curr})` : curr
        );
        
        return `${inputs[0]} = ${result}`;
    }
}

export class AllReduceOp extends ReduceOp {
    constructor(
        id: string,
        inShapes: number[][],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, inShapes, target, opType, params);
    }

    to_torch_functional(inputs: string[]): string {
        if (inputs.length < 1) {
            throw new Error("AllReduceOp requires at least 1 input");
        }

        const op = this._opType.toLowerCase();
        
        if (inputs.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        return `${inputs[0]} = torch.${op}(torch.stack([${inputs.join(", ")}], 0), 0)[0]`;
    }
} 