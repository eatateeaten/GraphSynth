// torch.abs(input)       # Absolute value
// torch.sqrt(input)      # Square root
// torch.square(input)    # Square
// torch.exp(input)       # Exponential
// torch.log(input)       # Natural logarithm
// torch.sin(input)
// torch.cos(input)
// torch.tan(input)
// torch.asin(input)
// torch.acos(input)
// torch.atan(inpu

//The Graph  
//Merge Node 
//Shape check for Merge and Branch 

import { getTorchCode } from './torch_nn_op_call_map';
import { getElementwiseOpCode } from './torch_nn_elementwise_call_map';

export abstract class GraphNode {
    protected readonly _id: string;
    protected _target: string;

    constructor(id: string, target: string) {
        // Validate UUID format
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        if (!uuidRegex.test(id)) {
            throw new Error(`Invalid UUID format: ${id}`);
        }
        this._id = id;
        this._target = target;
    }

    // Common getters
    get id(): string { return this._id; }
    get target(): string { return this._target; }

    //Static utility method for shape matching 
    //Gets called often, can be optimized more 
    static shapeMatch(shape1: number[], shape2: number[]): boolean {
        // Early return for null/undefined
        if (!shape1 || !shape2) return false;
        
        // Early return for different lengths
        if (shape1.length !== shape2.length) return false;
        
        // For small arrays, direct comparison is faster
        if (shape1.length <= 4) {
            return shape1.every((dim, i) => dim === shape2[i]);
        }
        
        // For larger arrays, use a more optimized approach
        // Using a for loop instead of every() for better performance
        for (let i = 0; i < shape1.length; i++) {
            if (shape1[i] !== shape2[i]) return false;
        }
        return true;
    }

    // Static utility method for index validation
    static validateIndex(index: number | undefined, arrayLength: number, context: string): number {
        if (index === undefined) {
            throw new Error(`${context}: index must be specified`);
        }
        
        if (index < 0 || index >= arrayLength) {
            throw new Error(`${context}: Invalid index ${index}. Must be between 0 and ${arrayLength - 1}`);
        }
        
        return index;
    }

    // Abstract methods that must be implemented by subclasses
    abstract to_torch_functional(input: string | string[]): string | string[];
    
    // Connection methods with optional index parameters
    abstract connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void; 
    abstract connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract disconnectSource(): void;
    abstract disconnectSink(): void; 

    // Add abstract shape getters
    abstract get inShape(): number[] | number[][];
    abstract get outShape(): number[] | number[][];

    // Add abstract connection getters/setters
    abstract get prev(): GraphNode | null;
    abstract set prev(node: GraphNode | null);
    abstract get next(): GraphNode | null;
    abstract set next(node: GraphNode | null);
}

export class Tensor extends GraphNode {
    protected _inShape: number[];
    protected _outShape: number[];
    private _prev: GraphNode | null;
    private _next: GraphNode | null; 

    constructor(
        id: string,
        Shape: number[],
        target: string
    ) {
        super(id, target);
        this._inShape = Shape;
        this._outShape = Shape; 
        this._prev = null;
        this._next = null;
    }

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[] { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    //!!!!!! there should be two dims for connect: from_dim and to_dim 
    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (prev instanceof Tensor) {
            this.connectTensor(prev);
        } else if (prev instanceof BranchOp) {
            prev.connectSink(this, indexPrev, undefined);
        } else if (prev instanceof Op || prev instanceof MergeOp) {
            // Now TypeScript knows prev.outShape is number[]
            if (!GraphNode.shapeMatch(this.inShape, prev.outShape)) {
                throw new Error(`Shape mismatch: Cannot connect tensor with shape [${this.inShape}] to prev with output shape [${prev.outShape}]`);
            }
            this._prev = prev;
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (next instanceof Tensor) {
            next.connectTensor(this);
        } else if (next instanceof MergeOp) {
            next.connectSource(this, indexNext, undefined);
        } else if (next instanceof Op || next instanceof BranchOp) {
            // Now TypeScript knows next.inShape is number[]
            if (!GraphNode.shapeMatch(this.outShape, next.inShape)) {
                throw new Error(`Shape mismatch: Cannot connect tensor with shape [${this.outShape}] to next with input shape [${next.inShape}]`);
            }
            this._next = next;
            next.prev = this;
        }
    }

    private connectTensor(tensor: Tensor): void {
        if (!GraphNode.shapeMatch(this.inShape, tensor.outShape)) {
            throw new Error(`Shape mismatch: Cannot connect tensors with shapes [${this.inShape}] and [${tensor.outShape}]`);
        }
        this._prev = tensor;
        tensor._next = this;
    }

    disconnectSource(): void {
        if (this._prev) {
            if (this._prev instanceof Tensor) {
                this._prev.next = null;
            } else if (this._prev instanceof BranchOp) {
                this._prev.disconnectSink(); //TODO implementation in BranchOp should take care of finding the index of this connection 
            } else if (this._prev instanceof Op || 
                       this._prev instanceof MergeOp) {
                this._prev.next = null;
            }
            this._prev = null;
        }
    }

    disconnectSink(): void {
        if (this._next) {
            if (this._next instanceof Tensor) {
                this._next.prev = null;
            } else if (this._next instanceof MergeOp) {
                this._next.disconnectSource();
            } else if (this._next instanceof Op || 
                       this._next instanceof BranchOp) {
                this._next.prev = null;
            }
            this._next = null;
        }
    }

    to_torch_functional(input: string): string {
        return input; // Tensor just passes through
    }
}

export class Op extends GraphNode {
    protected _inShape: number[];
    protected _outShape: number[];
    private _prev: GraphNode | null;
    private _next: GraphNode | null;
    private _opType: string;
    private _params: Record<string, any>;

    constructor(
        id: string,
        inShape: number[],
        outShape: number[],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, target);
        this._inShape = inShape;
        this._outShape = outShape;
        this._prev = null;
        this._next = null;
        this._opType = opType;
        this._params = params;
    }

    to_torch(): string { 
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        return getTorchCode(this._opType, this._params);
    }

    to_torch_functional(input: string): string { 
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        const module = this.to_torch();
        return `${input} = ${module}(${input})`;
    }

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[] { return this._outShape; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return this._next; }
    set next(node: GraphNode | null) { this._next = node; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (prev instanceof BranchOp) {
            prev.connectSink(this, indexPrev, undefined);
        } else {
            // At this point, prev can be either MergeOp, Tensor, or Op
            const prevShape = prev.outShape as number[];
            if (!GraphNode.shapeMatch(this._inShape, prevShape)) {
                throw new Error(`Shape mismatch: Cannot connect op with input shape [${this._inShape}] to prev with output shape [${prevShape}]`);
            }
            this._prev = prev;
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (next instanceof MergeOp) {
            next.connectSource(this, indexNext, undefined);
        } else {
            // At this point, next can be either BranchOp, Tensor, or Op
            const nextShape = next.inShape as number[];
            if (!GraphNode.shapeMatch(this._outShape, nextShape)) {
                throw new Error(`Shape mismatch: Cannot connect op with output shape [${this._outShape}] to next with input shape [${nextShape}]`);
            }
            this._next = next;
            next.prev = this;
        }
    }

    disconnectSource(): void {
        if (this._prev) {
            if (this._prev instanceof Tensor) {
                this._prev.next = null;
            } else if (this._prev instanceof BranchOp) {
                this._prev.disconnectSink();
            } else if (this._prev instanceof Op || 
                       this._prev instanceof MergeOp) {
                this._prev.next = null;
            }
            this._prev = null; 
        }
    }

    disconnectSink(): void {
        if (this._next) {
            if (this._next instanceof Tensor) {
                this._next.prev = null;
            } else if (this._next instanceof MergeOp) {
                this._next.disconnectSource();
            } else if (this._next instanceof Op || 
                       this._next instanceof BranchOp) {
                this._next.prev = null;
            }
            this._next = null; 
        }
    }
}

export abstract class MergeOp extends GraphNode {
    protected _inShapes: number[][];
    protected _outShape: number[];
    public _prevs: GraphNode[] = []; //Maybe a bad practice will change it later 
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
        this._outShape = this.computeOutShape(); //deal with this 
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

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        // Validate index using the static method
        const validatedIndex = GraphNode.validateIndex(indexSelf, this._inShapes.length, "MergeOp.connectSource");
        
        // Get the output shape from the previous node
        let prevOutShape: number[];
        
        if (prev instanceof Tensor || prev instanceof Op) {
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            // If a specific output of BranchOp is specified
            if (indexPrev !== undefined && indexPrev >= 0) {
                if (indexPrev >= prev.outShape.length) {
                    throw new Error(`Invalid sink index ${indexPrev} for BranchOp with ${prev.outShape.length} outputs`);
                }
                prevOutShape = prev.outShape[indexPrev];
            } else {
                // Default to first output if not specified
                prevOutShape = prev.outShape[0];
                indexPrev = 0;
            }
        } else if (prev instanceof MergeOp) {
            prevOutShape = prev.outShape;
        } else {
            throw new Error(`Cannot connect to node of type ${prev.constructor.name}`);
        }
        
        // Check shape compatibility
        if (!GraphNode.shapeMatch(this._inShapes[validatedIndex], prevOutShape)) {
            throw new Error(`Shape mismatch at index ${validatedIndex}: Cannot connect MergeOp with input shape [${this._inShapes[validatedIndex]}] to previous node with output shape [${prevOutShape}]`);
        }
        
        // Store the connection
        this._prevs[validatedIndex] = prev;
        
        // Create the reciprocal connection
        if (prev instanceof BranchOp && indexPrev !== undefined) {
            // Direct connection to BranchOp's nexts array
            prev._nexts[indexPrev] = this;
        } else {
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        // MergeOp has only one output
        if (this._next !== null) {
            throw new Error("MergeOp already has a sink connection");
        }
        
        // Check shape compatibility
        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof MergeOp) {
            // For MergeOp, we need to connect to a specific input
            if (indexNext === undefined) {
                indexNext = 0; // Default to first input if not specified
            }
            if (indexNext < 0 || indexNext >= next.inShape.length) {
                throw new Error(`Invalid source index ${indexNext} for MergeOp with ${next.inShape.length} inputs`);
            }
            nextInShape = next.inShape[indexNext];
        } else if (next instanceof BranchOp) {
            nextInShape = next.inShape;
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }
        
        // Check shape compatibility
        if (!GraphNode.shapeMatch(this._outShape, nextInShape)) {
            throw new Error(`Shape mismatch: Cannot connect MergeOp with output shape [${this._outShape}] to next node with input shape [${nextInShape}]`);
        }
        
        // Store the connection
        this._next = next;
        
        // Create the reciprocal connection
        if (next instanceof MergeOp && indexNext !== undefined) {
            // Direct connection to MergeOp's prevs array
            next._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(): void {
        // Remove all previous connections
        for (let i = 0; i < this._prevs.length; i++) {
            const prev = this._prevs[i];
            if (prev) {
                if (prev instanceof BranchOp) {
                    // Find our connection in BranchOp's nexts array and remove it
                    const index = prev._nexts.indexOf(this);
                    if (index >= 0) {
                        prev._nexts[index] = null as unknown as GraphNode;
                    }
                } else {
                    prev.next = null;
                }
                // Clear the reference with a cast to avoid type error
                this._prevs[i] = null as unknown as GraphNode;
            }
        }
    }

    disconnectSink(): void {
        if (this._next) {
            if (this._next instanceof MergeOp) {
                // Find our connection in MergeOp's prevs array and remove it
                const index = this._next._prevs.indexOf(this);
                if (index >= 0) {
                    this._next._prevs[index] = null as unknown as GraphNode;
                }
            } else {
                this._next.prev = null;
            }
            this._next = null;
        }
    }

    // Add the 'prev' getter/setter required by the abstract class
    get prev(): GraphNode | null {
        // For MergeOp, 'prev' typically refers to the first element in _prevs
        return this._prevs.length > 0 ? this._prevs[0] : null;
    }

    set prev(node: GraphNode | null) {
        // When setting 'prev', we clear existing sources and set only this one
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

        // Validate shapes
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

        // Compute output shape after validation
        const outShape = [...referenceShape];
        outShape[dim] = this._inShapes.reduce((sum, shape) => sum + shape[dim], 0);
        return outShape;
    }

    to_torch_functional(inputs: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }

    // Add inherited abstract getter/setter
    get prev(): GraphNode | null {
        return super.prev;
    }
    
    set prev(node: GraphNode | null) {
        super.prev = node;
    }
}

export class ElementwiseOp extends MergeOp {
    constructor(id: string, inShapes: number[][], target: string, opType: string) {
        super(id, inShapes, target, opType, {});
    }

    protected computeOutShape(): number[] {
        if (this._inShapes.length < 2) {
            throw new Error("ElementwiseOp requires at least 2 input tensors");
        }

        // Start with the first shape
        let resultShape = [...this._inShapes[0]];

        // Check each subsequent shape for broadcast compatibility
        for (let i = 1; i < this._inShapes.length; i++) {
            const currentShape = this._inShapes[i];
            
            // Validate ranks
            if (currentShape.length !== resultShape.length) {
                throw new Error(`Input shapes must have the same rank. Shape at index ${i} has rank ${currentShape.length}, expected ${resultShape.length}`);
            }
            
            //TODO check this: might have an error here 
            // For each dimension, the output will be the maximum of the two sizes
            // unless one of them is 1 (broadcasting)
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
            throw new Error("ElementwiseOp requires at least 2 inputs");
        }
        const opCode = getElementwiseOpCode(this._opType);
        
        // Use reduce to accumulate the nested function calls
        const result = inputs.reduce((acc, curr) => 
            acc ? `${opCode}(${acc}, ${curr})` : curr
        );
        
        return `${inputs[0]} = ${result}`;
    }

    // Add inherited abstract getter/setter
    get prev(): GraphNode | null {
        return super.prev;
    }
    
    set prev(node: GraphNode | null) {
        super.prev = node;
    }
}

export class DotOp extends MergeOp {
    constructor(id: string, inShapes: number[][], target: string, opType: string, params: { dim: number }) {
        super(id, inShapes, target, opType, params);
    }

    protected computeOutShape(): number[] {
        throw new Error("Not implemented");
    }

    to_torch_functional(inputs: string[]): string {
        throw new Error("Not implemented");
    }

    // Add inherited abstract getter/setter
    get prev(): GraphNode | null {
        return super.prev;
    }
    
    set prev(node: GraphNode | null) {
        super.prev = node;
    }
}

export class CrossOp extends MergeOp {
    constructor(id: string, inShapes: number[][], target: string, opType: string, params: { dim: number }) {
        super(id, inShapes, target, opType, params);
    }

    protected computeOutShape(): number[] {
        throw new Error("Not implemented");
    }

    to_torch_functional(inputs: string[]): string {
        throw new Error("Not implemented");
    }

    // Add inherited abstract getter/setter
    get prev(): GraphNode | null {
        return super.prev;
    }
    
    set prev(node: GraphNode | null) {
        super.prev = node;
    }
}

export abstract class BranchOp extends GraphNode {
    protected _inShape: number[];
    protected _outShapes: number[][];
    protected _prev: GraphNode | null = null;
    public _nexts: GraphNode[] = [];
    protected readonly _opType: string;
    protected readonly _params: Record<string, any>;

    constructor(
        id: string,
        inShape: number[],
        target: string,
        opType: string,
        params: Record<string, any>
    ) {
        super(id, target);
        this._inShape = inShape;
        this._opType = opType;
        this._params = params;
        this._outShapes = this.computeOutShapes();
    }

    protected abstract computeOutShapes(): number[][];
    abstract to_torch_functional(input: string): string[];

    // Getters and setters
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[][] { return this._outShapes; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get nexts(): GraphNode[] { return [...this._nexts]; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }

    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        // BranchOp has only one input
        if (this._prev !== null) {
            throw new Error("BranchOp already has a source connection");
        }
        
        // Get the output shape from the previous node
        let prevOutShape: number[];
        if (prev instanceof Tensor || prev instanceof Op) {
            prevOutShape = prev.outShape as number[];
        } else if (prev instanceof BranchOp) {
            // If a specific output of BranchOp is specified
            if (indexPrev !== undefined && indexPrev >= 0) {
                if (indexPrev >= prev.outShape.length) {
                    throw new Error(`Invalid sink index ${indexPrev} for BranchOp with ${prev.outShape.length} outputs`);
                }
                prevOutShape = prev.outShape[indexPrev];
            } else {
                // Default to first output if not specified
                prevOutShape = prev.outShape[0];
                indexPrev = 0;
            }
        } else if (prev instanceof MergeOp) {
            prevOutShape = prev.outShape;
        } else {
            throw new Error(`Cannot connect to node of type ${prev.constructor.name}`);
        }
        
        // Check shape compatibility
        if (!GraphNode.shapeMatch(this._inShape, prevOutShape)) {
            throw new Error(`Shape mismatch: Cannot connect BranchOp with input shape [${this._inShape}] to previous node with output shape [${prevOutShape}]`);
        }
        
        // Store the connection
        this._prev = prev;
        
        // Create the reciprocal connection
        if (prev instanceof BranchOp) {
            // For BranchOp, use the setConnectionAt method to set a specific index
            if (indexPrev !== undefined && indexPrev >= 0) {
                prev.setConnectionAt(indexPrev, this);
            } else {
                prev.next = this;
            }
        } else {
            prev.next = this;
        }
    }

    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        // Validate index using the static method
        const validatedIndex = GraphNode.validateIndex(indexSelf, this._outShapes.length, "BranchOp.connectSink");
        
        // Check shape compatibility
        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof MergeOp) {
            // For MergeOp, we need to connect to a specific input
            if (indexNext === undefined) {
                indexNext = 0; // Default to first input if not specified
            }
            if (indexNext < 0 || indexNext >= next.inShape.length) {
                throw new Error(`Invalid source index ${indexNext} for MergeOp with ${next.inShape.length} inputs`);
            }
            nextInShape = next.inShape[indexNext];
        } else if (next instanceof BranchOp) {
            nextInShape = next.inShape;
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }
        
        // Check if our output shape matches the next node's input shape
        if (!GraphNode.shapeMatch(this._outShapes[validatedIndex], nextInShape)) {
            throw new Error(`Shape mismatch at index ${validatedIndex}: Cannot connect BranchOp with output shape [${this._outShapes[validatedIndex]}] to next node with input shape [${nextInShape}]`);
        }
        
        // Store the connection
        this._nexts[validatedIndex] = next;
        
        // Create the reciprocal connection
        if (next instanceof MergeOp && indexNext !== undefined) {
            // Direct connection to MergeOp's prevs array
            next._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(): void {
        if (this._prev) {
            if (this._prev instanceof BranchOp) {
                // Find our connection in BranchOp's nexts array and remove it
                const index = this._prev._nexts.indexOf(this);
                if (index >= 0) {
                    this._prev._nexts[index] = null as unknown as GraphNode;
                }
            } else {
                this._prev.next = null;
            }
            this._prev = null;
        }
    }

    disconnectSink(): void {
        // Remove all next connections
        for (let i = 0; i < this._nexts.length; i++) {
            const next = this._nexts[i];
            if (next) {
                if (next instanceof MergeOp) {
                    // Find our connection in MergeOp's prevs array and remove it
                    const index = next._prevs.indexOf(this);
                    if (index >= 0) {
                        next._prevs[index] = null as unknown as GraphNode;
                    }
                } else {
                    next.prev = null;
                }
                // Clear the reference with a cast to avoid type error
                this._nexts[i] = null as unknown as GraphNode;
            }
        }
    }

    // Add the 'next' getter/setter required by the abstract class
    get next(): GraphNode | null {
        // For BranchOp, 'next' typically refers to the first element in _nexts
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        // When setting 'next', we clear existing sinks and set only this one
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }

    // Add a method to set a specific connection at index for MergeOp to use
    setConnectionAt(index: number, node: GraphNode): void {
        if (index >= 0 && index < this._outShapes.length) {
            this._nexts[index] = node;
        } else {
            throw new Error(`Invalid index ${index} for BranchOp with ${this._outShapes.length} outputs`);
        }
    }

    // Add a method to directly set a connection at a specific index
    setNextAt(index: number, node: GraphNode): void {
        if (index >= 0 && index < this._outShapes.length) {
            this._nexts[index] = node;
        } else {
            throw new Error(`Invalid index ${index} for BranchOp with ${this._outShapes.length} outputs`);
        }
    }
}

export class Split extends BranchOp {
    constructor(id: string, inShape: number[], target: string, params: { sections: number[], dim: number }) {
        super(id, inShape, target, "Split", params);
    }

    protected computeOutShapes(): number[][] {
        return this._params.sections.map((size: number) => {
            const shape = [...this._inShape];
            shape[this._params.dim] = size;
            return shape;
        });
    }

    to_torch_functional(input: string): string[] {
        const outputs = this._outShapes.map((_, i) => `out${i}`);
        return [`${outputs.join(', ')} = torch.split(${input}, [${this._params.sections.join(', ')}], dim=${this._params.dim})`];
    }

    // Add inherited abstract getter/setter
    get next(): GraphNode | null {
        return super.next;
    }
    
    set next(node: GraphNode | null) {
        super.next = node;
    }
}

export class Copy extends BranchOp {
    constructor(id: string, inShape: number[], target: string, params: { copies: number }) {
        super(id, inShape, target, "Copy", params);
    }

    protected computeOutShapes(): number[][] {
        return Array(this._params.copies).fill(0).map(() => [...this._inShape]);
    }

    to_torch_functional(input: string): string[] {
        return Array(this._params.copies).fill(0).map((_, i) => `out${i} = ${input}`);
    }

    // Add inherited abstract getter/setter
    get next(): GraphNode | null {
        return super.next;
    }
    
    set next(node: GraphNode | null) {
        super.next = node;
    }
}






