// export class CheckerGraph {
//     private nodes = new Map<string, CheckerNode<any>>();

//     addNode<T extends NodeParams>(id: string, node: CheckerNode<T>): void {
//         this.nodes.set(id, node);
//     }

//     getNode(id: string): CheckerNode<any> | undefined {
//         return this.nodes.get(id);
//     }

//     connect(sourceId: string, targetId: string): void {
//         const source = this.nodes.get(sourceId);
//         const target = this.nodes.get(targetId);
        
//         if (!source || !target) {
//             throw new Error("Node not found");
//         }

//         source.connectTo(target);
//     }

//     deleteNode(id: string): void {
//         const node = this.nodes.get(id);
//         if (!node) return;

//         if (node.in_node) {
//             node.in_node.out_node = null;
//         }
//         if (node.out_node) {
//             node.out_node.in_node = null;
//         }

//         this.nodes.delete(id);
//     }
// }



import { v4 as uuidv4 } from 'uuid';
import { getTorchCode } from './torch_nn_op_call_map';
import { getPointwiseOpCode } from './torch_nn_pointwise_call_map';

export class Tensor {
    //Anchors for the training graph 
    private readonly _id: string;
    protected _inShape: number[];
    protected _outShape: number[];
    private _prev: Tensor | Op | null;
    private _next: Tensor | Op | null;
    private _target: string;

    constructor(
        inShape: number[],
        outShape: number[],
        target: string
    ) {
        this._id = uuidv4();
        this._inShape = inShape;
        this._outShape = outShape;
        this._prev = null;
        this._next = null;
        this._target = target;
    }

    // Getter for id
    get id(): string {
        return this._id;
    }

    // Getters
    get inShape(): number[] {
        return this._inShape;
    }

    get outShape(): number[] {
        return this._outShape;
    }

    get prev(): Tensor | Op | null {
        return this._prev;
    }

    get next(): Tensor | Op | null {
        return this._next;
    }

    get target(): string {
        return this._target;
    }

    // Setters
    set prev(node: Tensor | Op | null) {
        this._prev = node;
    }

    set next(node: Tensor | Op | null) {
        this._next = node;
    }
}

export class Op {
    private readonly _id: string;
    protected _inShape: number[];
    protected _outShape: number[];
    private _prev: Op | null;
    private _next: Op | null;
    private _target: string;
    private _opType: string;
    private _params: Record<string, any>;

    constructor(
        inShape: number[],
        outShape: number[],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        this._id = uuidv4();
        this._inShape = inShape;
        this._outShape = outShape;
        this._prev = null;
        this._next = null;
        this._target = target;
        this._opType = opType;
        this._params = params;
    }

    to_torch(): string { 
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        } 
        //TODO We need to add the dim_select stuff here otherwise it wouldn't work for some PyTorch nn.module lol lol 
        return getTorchCode(this._opType, this._params);
    }

    to_torch_functional(input: string): string { //Note: user needs to make sure the input string is a legit python variable name 
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        const module = this.to_torch();
        return `${input} = ${module}(${input})`;
    }

    // Getter for id
    get id(): string {
        return this._id;
    }

    // Getters
    get inShape(): number[] {
        return this._inShape;
    }

    get outShape(): number[] {
        return this._outShape;
    }

    get prev(): Op | null {
        return this._prev;
    }

    get next(): Op | null {
        return this._next;
    }

    get target(): string {
        return this._target;
    }

    get opType(): string {
        return this._opType;
    }

    get params(): Record<string, any> {
        return { ...this._params };
    }

    // Setters
    set prev(op: Op | null) {
        this._prev = op;
    }

    set next(op: Op | null) {
        this._next = op;
    }
}

export class Seq extends Op implements Iterable<Op> {
    private _operations: Op[];

    constructor(initialOp: Op) {
        super(
            initialOp.inShape,
            initialOp.outShape,
            initialOp.target,
            "Seq",
            {}
        );
        this._operations = [initialOp];
    }

    to_torch(): string {
        // Verify all operations are PyTorch operations
        for (const op of this._operations) {
            if (op.target !== "torch") {
                throw new Error("All operations in sequence must be PyTorch operations");
            }
        }

        // Generate sequential module code
        const moduleLines = this._operations.map((op, index) => {
            return `            (${index}): ${op.to_torch()}`;
        });

        return `nn.Sequential(\n${moduleLines.join(',\n')}\n        )`;
    }

    to_torch_functional(input: string): string {
        if (this._operations.some(op => op.target !== "torch")) {
            throw new Error("All operations in sequence must be PyTorch operations");
        }

        // Generate multi-line code with each operation on its own line
        const lines = this._operations.map(op => {
            return `        ${op.to_torch_functional(input)}`;
        });

        return lines.join('\n');
    }

    private shapeMatch(op1: Op, op2: Op): boolean {
        if (!op1.outShape || !op2.inShape) {
            return false;
        }
        
        if (op1.outShape.length !== op2.inShape.length) {
            return false;
        }

        return op1.outShape.every((dim, index) => dim === op2.inShape[index]);
    }

    findById(id: string): Op | undefined {
        return this._operations.find(op => op.id === id);
    }

    push(op: Op): string {
        // Check shape compatibility before any modifications
        if (this._operations.length > 0) {
            const lastOp = this._operations[this._operations.length - 1];
            if (!this.shapeMatch(lastOp, op)) {
                throw new Error(`Shape mismatch: Cannot connect output shape [${lastOp.outShape}] to input shape [${op.inShape}]`);
            }
            // Set up links only after shape check passes
            lastOp.next = op;
            op.prev = lastOp;
        }

        // Add to array only after all validations pass
        this._operations.push(op);
        
        // Update Seq's shapes
        this._outShape = op.outShape;
        
        return op.id;
    }

    pop(): Op | undefined {
        if (this._operations.length <= 1) {
            throw new Error("Cannot pop from sequence with only one operation");
        }

        const lastOp = this._operations[this._operations.length - 1];
        // Get the new last op's outShape before removing
        const newLastOp = this._operations[this._operations.length - 2];
        
        if (this.remove(lastOp.id)) {
            // Update outShape to the new last operation's outShape
            this._outShape = newLastOp.outShape;
            return lastOp;
        }
        return undefined;
    }

    insert(op: Op, index: number): string {
        if (index < 0 || index > this._operations.length) {
            throw new Error("Index out of bounds");
        }

        if (index === this._operations.length) {
            return this.push(op);
        }

        // Check shape compatibility with previous op
        const prevOp = index > 0 ? this._operations[index - 1] : null;
        const nextOp = this._operations[index];

        if (prevOp && !this.shapeMatch(prevOp, op)) {
            throw new Error(`Shape mismatch: Cannot connect output shape [${prevOp.outShape}] to input shape [${op.inShape}]`);
        }

        // Check shape compatibility with next op
        if (!this.shapeMatch(op, nextOp)) {
            throw new Error(`Shape mismatch: Cannot connect output shape [${op.outShape}] to input shape [${nextOp.inShape}]`);
        }

        // Update links after shape checks pass
        if (prevOp) {
            prevOp.next = op;
            op.prev = prevOp;
        }

        op.next = nextOp;
        nextOp.prev = op;

        // Insert into array
        this._operations.splice(index, 0, op);

        // Update Seq's shapes if inserting at the beginning
        if (index === 0) {
            this._inShape = op.inShape;
        }
        
        return op.id;
    }

    remove(id: string): boolean {
        if (this._operations.length <= 1) {
            throw new Error("Cannot remove from sequence with only one operation");
        }

        const opIndex = this._operations.findIndex(op => op.id === id);
        if (opIndex === -1) {
            return false;  // Operation not found
        }

        const op = this._operations[opIndex];
        const prevOp = op.prev;
        const nextOp = op.next;
        
        // Check if the remaining operations can be connected
        if (prevOp && nextOp && !this.shapeMatch(prevOp, nextOp)) {
            throw new Error(`Shape mismatch: Cannot remove op as it would create invalid connection between output shape [${prevOp.outShape}] and input shape [${nextOp.inShape}]`);
        }

        // Update links after shape check passes
        if (prevOp) {
            prevOp.next = nextOp;
        }
        if (nextOp) {
            nextOp.prev = prevOp;
        }

        // Remove from array
        this._operations.splice(opIndex, 1);

        // Update Seq's shapes
        if (opIndex === 0) {
            // If removing first operation, update inShape
            this._inShape = this._operations[0].inShape;
        }
        if (opIndex === this._operations.length - 1) {
            // If removing last operation, update outShape
            this._outShape = this._operations[this._operations.length - 1].outShape;
        }

        return true;
    }

    [Symbol.iterator](): Iterator<Op> {
        let index = 0;
        
        return {
            next: (): IteratorResult<Op> => {
                if (index < this._operations.length) {
                    return {
                        value: this._operations[index++],
                        done: false
                    };
                } else {
                    return {
                        value: null as any,
                        done: true
                    };
                }
            }
        };
    }

    // Additional helper methods
    get length(): number {
        return this._operations.length;
    }

    get operations(): Op[] {
        return [...this._operations];
    }
}


export interface MergeOp {
    // Getters
    get id(): string;
    get inShapes(): number[][];
    get outShape(): number[];
    get prevs(): Tensor[];
    get next(): Op | Tensor | null;
    get target(): string;
    get opType(): string;
    get params(): Record<string, any>;

    // Setters
    set next(node: Op | Tensor | null);

    // Methods
    to_torch_functional(inputs: string[]): string;
    //addPrev(tensor: Tensor): void;
}

export interface BranchOp {
    // Getters
    get id(): string;
    get inShape(): number[];
    get outShapes(): number[][];
    get prev(): Tensor | Op | null;
    get nexts(): Tensor[];
    get target(): string;
    get opType(): string;
    get params(): Record<string, any>;

    // Setters
    set prev(node: Tensor | Op | null);

    // Methods
    to_torch_functional(input: string): string[];
    //addNext(tensor: Tensor): void;
}

export class ConcatMerge implements MergeOp {
    private readonly _id: string;
    protected _inShapes: number[][];
    protected _outShape: number[];
    private _prevs: Tensor[] = [];
    private _next: Op | Tensor | null = null;
    private readonly _target: string;
    private readonly _opType: string = "Concat";
    private readonly _params: Record<string, any>;

    constructor(inShapes: number[][], target: string, params: { dim: number }) {
        const dim = params.dim;
        if (dim < 0 || dim >= inShapes[0].length) {
            throw new Error(`Invalid concatenation dimension ${dim} for input shape of length ${inShapes[0].length}`);
        }

        // Check that all input shapes have the same dimensions except at concat_dim
        const referenceShape = inShapes[0];
        for (let i = 1; i < inShapes.length; i++) {
            const shape = inShapes[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For concatenation, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (j !== dim && shape[j] !== referenceShape[j]) {
                    throw new Error(`For concatenation, input shapes must match on all dimensions except the concatenation dimension. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        this._id = uuidv4();
        this._inShapes = inShapes;
        this._target = target;
        this._params = params;

        // Calculate output shape
        this._outShape = [...referenceShape];
        this._outShape[dim] = inShapes.reduce((sum, shape) => sum + shape[dim], 0);
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShapes(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get prevs(): Tensor[] { return [...this._prevs]; }
    get next(): Op | Tensor | null { return this._next; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set next(node: Op | Tensor | null) { this._next = node; }

    to_torch_functional(inputs: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }

    // addPrev(tensor: Tensor): void {
    //     if (this._prevs.length >= this._inShapes.length) {
    //         throw new Error("Cannot add more inputs than specified in inShapes");
    //     }
    //     const shapeIndex = this._prevs.length;
    //     if (!this.shapeMatch(tensor.outShape, this._inShapes[shapeIndex])) {
    //         throw new Error(`Shape mismatch: Cannot connect output shape [${tensor.outShape}] to input shape [${this._inShapes[shapeIndex]}]`);
    //     }
    //     this._prevs.push(tensor);
    // }
}

export class PointwiseMerge implements MergeOp {
    private readonly _id: string;
    protected _inShapes: number[][];
    protected _outShape: number[];
    private _prevs: Tensor[] = [];
    private _next: Op | Tensor | null = null;
    private readonly _target: string;
    private readonly _opType: string;
    private readonly _params: Record<string, any>;

    constructor(inShapes: number[][], target: string, opType: string) {
        // Check that all input shapes are identical
        const referenceShape = inShapes[0];
        for (let i = 1; i < inShapes.length; i++) {
            const shape = inShapes[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For pointwise operations, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (shape[j] !== referenceShape[j]) {
                    throw new Error(`For pointwise operations, all input shapes must be identical. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        this._id = uuidv4();
        this._inShapes = inShapes;
        this._target = target;
        this._opType = opType;
        this._params = {};
        this._outShape = [...referenceShape];
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShapes(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get prevs(): Tensor[] { return [...this._prevs]; }
    get next(): Op | Tensor | null { return this._next; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set next(node: Op | Tensor | null) { this._next = node; }

    to_torch_functional(inputs: string[]): string {
        throw new Error("Not implemented");
    }

    // addPrev(tensor: Tensor): void {
    //     if (this._prevs.length >= this._inShapes.length) {
    //         throw new Error("Cannot add more inputs than specified in inShapes");
    //     }
    //     if (!this.shapeMatch(tensor.outShape, this._outShape)) {
    //         throw new Error(`Shape mismatch: Cannot connect output shape [${tensor.outShape}] to required shape [${this._outShape}]`);
    //     }
    //     this._prevs.push(tensor);
    // }
}

export class DotMerge implements MergeOp {
    private readonly _id: string;
    protected _inShapes: number[][];
    protected _outShape: number[];
    private _prevs: Tensor[] = [];
    private _next: Op | Tensor | null = null;
    private readonly _target: string;
    private readonly _opType: string;
    private readonly _params: Record<string, any>;

    constructor(inShapes: number[][], target: string, opType: string, params: { dim: number }) {
        const dim = params.dim;
        if (dim < 0 || dim >= inShapes[0].length) {
            throw new Error(`Invalid dimension ${dim} for input shape of length ${inShapes[0].length}`);
        }

        // Check that all input shapes have the same dimensions
        const referenceShape = inShapes[0];
        for (let i = 1; i < inShapes.length; i++) {
            const shape = inShapes[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For dot operations, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (shape[j] !== referenceShape[j]) {
                    throw new Error(`For dot operations, all input shapes must be identical. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        this._id = uuidv4();
        this._inShapes = inShapes;
        this._target = target;
        this._opType = opType;
        this._params = params;

        // Output shape is input shape with the dot product dimension removed
        this._outShape = [...referenceShape];
        this._outShape.splice(dim, 1);
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShapes(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get prevs(): Tensor[] { return [...this._prevs]; }
    get next(): Op | Tensor | null { return this._next; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set next(node: Op | Tensor | null) { this._next = node; }

    to_torch_functional(inputs: string[]): string {
        throw new Error("Not implemented");
    }
}

export class CrossMerge implements MergeOp {
    private readonly _id: string;
    protected _inShapes: number[][];
    protected _outShape: number[];
    private _prevs: Tensor[] = [];
    private _next: Op | Tensor | null = null;
    private readonly _target: string;
    private readonly _opType: string;
    private readonly _params: Record<string, any>;

    constructor(inShapes: number[][], target: string, opType: string, params: { dim: number }) {
        const dim = params.dim;
        if (dim < 0 || dim >= inShapes[0].length) {
            throw new Error(`Invalid dimension ${dim} for input shape of length ${inShapes[0].length}`);
        }

        // Check that all input shapes have the same dimensions
        const referenceShape = inShapes[0];
        for (let i = 1; i < inShapes.length; i++) {
            const shape = inShapes[i];
            if (shape.length !== referenceShape.length) {
                throw new Error(`For cross operations, all input shapes must have the same rank. Shape at index ${i} has rank ${shape.length}, expected ${referenceShape.length}`);
            }
            for (let j = 0; j < shape.length; j++) {
                if (shape[j] !== referenceShape[j]) {
                    throw new Error(`For cross operations, all input shapes must be identical. Mismatch at shape index ${i}, dimension ${j}: got ${shape[j]}, expected ${referenceShape[j]}`);
                }
            }
        }

        this._id = uuidv4();
        this._inShapes = inShapes;
        this._target = target;
        this._opType = opType;
        this._params = params;

        // Output shape is same as input shape with the cross operation dimension potentially modified
        this._outShape = [...referenceShape];
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShapes(): number[][] { return this._inShapes; }
    get outShape(): number[] { return this._outShape; }
    get prevs(): Tensor[] { return [...this._prevs]; }
    get next(): Op | Tensor | null { return this._next; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set next(node: Op | Tensor | null) { this._next = node; }

    to_torch_functional(inputs: string[]): string {
        throw new Error("Not implemented");
    }
}

export class SplitBranch implements BranchOp {
    private readonly _id: string;
    protected _inShape: number[];
    protected _outShapes: number[][];
    private _prev: Tensor | Op | null = null;
    private _nexts: Tensor[] = [];
    private readonly _target: string;
    private readonly _opType: string = "Split";
    private readonly _params: Record<string, any>;

    constructor(inShape: number[], target: string, params: { sections: number[], dim: number }) {
        this._id = uuidv4();
        this._inShape = inShape;
        this._target = target;
        this._params = params;

        this._outShapes = params.sections.map(size => {
            const shape = [...inShape];
            shape[params.dim] = size;
            return shape;
        });
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShape(): number[] { return this._inShape; }
    get outShapes(): number[][] { return this._outShapes; }
    get prev(): Tensor | Op | null { return this._prev; }
    get nexts(): Tensor[] { return [...this._nexts]; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set prev(node: Tensor | Op | null) { this._prev = node; }

    // Implement methods
    to_torch(): string {
        return `torch.split(split_size_or_sections=[${this._params.sections.join(', ')}], dim=${this._params.dim})`;
    }

    to_torch_functional(input: string): string[] {
        const outputs = this._outShapes.map((_, i) => `out${i}`);
        return [`${outputs.join(', ')} = torch.split(${input}, [${this._params.sections.join(', ')}], dim=${this._params.dim})`];
    }

    // addNext(tensor: Tensor): void {
    //     if (this._nexts.length >= this._outShapes.length) {
    //         throw new Error("Cannot add more outputs than specified splits");
    //     }
    //     const shapeIndex = this._nexts.length;
    //     if (!this.shapeMatch(this._outShapes[shapeIndex], tensor.inShape)) {
    //         throw new Error(`Shape mismatch: Cannot connect output shape [${this._outShapes[shapeIndex]}] to input shape [${tensor.inShape}]`);
    //     }
    //     this._nexts.push(tensor);
    // }
}

export class CopyBranch implements BranchOp {
    private readonly _id: string;
    protected _inShape: number[];
    protected _outShapes: number[][];
    private _prev: Tensor | Op | null = null;
    private _nexts: Tensor[] = [];
    private readonly _target: string;
    private readonly _opType: string = "Copy";
    private readonly _params: Record<string, any>;

    constructor(inShape: number[], target: string, params: { copies: number }) {
        this._id = uuidv4();
        this._inShape = inShape;
        this._target = target;
        this._params = params;

        this._outShapes = [];
        for (let i = 0; i < params.copies; i++) {
            this._outShapes.push([...inShape]);
        }
    }

    // Implement getters
    get id(): string { return this._id; }
    get inShape(): number[] { return this._inShape; }
    get outShapes(): number[][] { return this._outShapes; }
    get prev(): Tensor | Op | null { return this._prev; }
    get nexts(): Tensor[] { return [...this._nexts]; }
    get target(): string { return this._target; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    // Implement setter
    set prev(node: Tensor | Op | null) { this._prev = node; }

    // Implement methods
    to_torch(): string {
        return "copy";
    }

    to_torch_functional(input: string): string[] {
        const result: string[] = [];
        for (let i = 0; i < this._params.copies; i++) {
            result.push(`out${i} = ${input}.clone()`);
        }
        return result;
    }

    // addNext(tensor: Tensor): void {
    //     if (this._nexts.length >= this._params.copies) {
    //         throw new Error("Cannot add more outputs than specified copies");
    //     }
    //     if (!this.shapeMatch(this._inShape, tensor.inShape)) {
    //         throw new Error(`Shape mismatch: Cannot connect output shape [${this._inShape}] to input shape [${tensor.inShape}]`);
    //     }
    //     this._nexts.push(tensor);
    // }
}




