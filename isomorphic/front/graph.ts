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
        if (!shape1 || !shape2) return false;
        
        if (shape1.length !== shape2.length) return false;
        
        if (shape1.length <= 4) {
            return shape1.every((dim, i) => dim === shape2[i]);
        }
        
        // For larger arrays, 
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
    abstract disconnectSource(indexSelf?: number): void;
    abstract disconnectSink(indexSelf?: number): void;

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
            if (indexPrev === undefined) { 
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
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
            if (indexNext === undefined) { //Change this for ReduceOp
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
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

    disconnectSource(indexSelf?: number): void {
        if (this._prev) {
            if (this._prev instanceof Tensor) {
                this._prev.next = null;
            } else if (this._prev instanceof BranchOp) {
                // Find which output of the BranchOp we're connected to
                const index = this._prev._nexts.indexOf(this);
                if (index >= 0) {
                    this._prev._nexts[index] = null as unknown as GraphNode;
                }
            } else if (this._prev instanceof Op || 
                       this._prev instanceof MergeOp) {
                this._prev.next = null;
            }
            this._prev = null;
        }
    }

    disconnectSink(indexSelf?: number): void {
        if (this._next) {
            if (this._next instanceof Tensor) {
                this._next.prev = null;
            } else if (this._next instanceof MergeOp) {
                // Find which input of the MergeOp we're connected to
                const index = this._next._prevs.indexOf(this);
                if (index >= 0) {
                    this._next._prevs[index] = null as unknown as GraphNode;
                }
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
            if (indexPrev === undefined) {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
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
            if (indexNext === undefined) { //TODO change this for ReduceOp
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
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

    disconnectSource(indexSelf?: number): void {
        // Op has only one input, so indexSelf is ignored
        if (this._prev) {
            if (this._prev instanceof Tensor) {
                this._prev.next = null;
            } else if (this._prev instanceof BranchOp) {
                // Find which output of the BranchOp we're connected to
                const index = this._prev._nexts.indexOf(this);
                if (index >= 0) {
                    this._prev._nexts[index] = null as unknown as GraphNode;
                }
            } else if (this._prev instanceof Op || 
                       this._prev instanceof MergeOp) {
                this._prev.next = null;
            }
            this._prev = null; 
        }
    }

    disconnectSink(indexSelf?: number): void {
        // Op has only one output, so indexSelf is ignored
        if (this._next) {
            if (this._next instanceof Tensor) {
                this._next.prev = null;
            } else if (this._next instanceof MergeOp) {
                // Find which input of the MergeOp we're connected to
                const index = this._next._prevs.indexOf(this);
                if (index >= 0) {
                    this._next._prevs[index] = null as unknown as GraphNode;
                }
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
            if (indexPrev !== undefined) {
                // Validate BranchOp output index
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "MergeOp.connectSource (BranchOp output)");
                prevOutShape = prev.outShape[validatedPrevIndex];
                indexPrev = validatedPrevIndex;
            } else {
                // Require explicit index for BranchOp outputs
                throw new Error("When connecting from a BranchOp, an output index must be specified");
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
            if (indexNext === undefined) { //TODO change this for ReduceOp 
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            // Validate MergeOp input index
            const validatedNextIndex = GraphNode.validateIndex(indexNext, next.inShape.length, "MergeOp.connectSink (MergeOp input)");
            nextInShape = next.inShape[validatedNextIndex];
            indexNext = validatedNextIndex;
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

    disconnectSource(indexSelf?: number): void {
        if (indexSelf !== undefined) {
            // Disconnect a specific input
            const validatedIndex = GraphNode.validateIndex(indexSelf, this._inShapes.length, "MergeOp.disconnectSource");
            this._disconnectSourceAtIndex(validatedIndex);
        } else {
            // Disconnect all inputs
        for (let i = 0; i < this._prevs.length; i++) {
                if (this._prevs[i]) {
                    this._disconnectSourceAtIndex(i);
                }
            }
        }
    }

    private _disconnectSourceAtIndex(index: number): void {
        const prev = this._prevs[index];
            if (prev) {
                if (prev instanceof BranchOp) {
                    // Find our connection in BranchOp's nexts array and remove it
                const branchIndex = prev._nexts.indexOf(this);
                if (branchIndex >= 0) {
                    prev._nexts[branchIndex] = null as unknown as GraphNode;
                    }
                } else {
                    prev.next = null;
                }
                // Clear the reference with a cast to avoid type error
            this._prevs[index] = null as unknown as GraphNode;
        }
    }

    disconnectSink(indexSelf?: number): void {
        // MergeOp has only one output, so indexSelf is ignored
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
        } else if (prev instanceof MergeOp) {
            prevOutShape = prev.outShape;
        } else if (prev instanceof BranchOp) {
            // If a specific output of BranchOp is specified
            if (indexPrev !== undefined) {
                // Validate BranchOp output index
                const validatedPrevIndex = GraphNode.validateIndex(indexPrev, prev.outShape.length, "BranchOp.connectSource (BranchOp output)");
                prevOutShape = prev.outShape[validatedPrevIndex];
                indexPrev = validatedPrevIndex;
            } else {
                // Require explicit index for BranchOp outputs // Change for MapOp
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
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
                prev._nexts[indexPrev] = this;
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
        
        // Get next's inShape 
        let nextInShape: number[];
        if (next instanceof Tensor || next instanceof Op) {
            nextInShape = next.inShape as number[];
        } else if (next instanceof BranchOp) {
            nextInShape = next.inShape;
        } else if (next instanceof MergeOp) {
            // For MergeOp, we need to connect to a specific input
            if (indexNext === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            // Validate MergeOp input index
            const validatedNextIndex = GraphNode.validateIndex(indexNext, next.inShape.length, "BranchOp.connectSink (MergeOp input)");
            nextInShape = next.inShape[validatedNextIndex];
            indexNext = validatedNextIndex;
        } else {
            throw new Error(`Cannot connect to node of type ${next.constructor.name}`);
        }
        
        // Check if our output shape matches the next node's input shape
        if (!GraphNode.shapeMatch(this._outShapes[validatedIndex], nextInShape)) {
            throw new Error(`Shape mismatch at index ${validatedIndex}: Cannot connect BranchOp with output shape [${this._outShapes[validatedIndex]}] to next node with input shape [${nextInShape}]`);
        }
        
        // Create the connection
        this._nexts[validatedIndex] = next;
        
        // Create the reciprocal connection
        if (next instanceof MergeOp && indexNext !== undefined) {
            // Direct connection to MergeOp's prevs array
            next._prevs[indexNext] = this;
        } else {
            next.prev = this;
        }
    }

    disconnectSource(indexSelf?: number): void {
        // BranchOp has only one input, so indexSelf is ignored
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

    disconnectSink(indexSelf?: number): void {
        if (indexSelf !== undefined) {
            // Disconnect a specific output
            const validatedIndex = GraphNode.validateIndex(indexSelf, this._outShapes.length, "BranchOp.disconnectSink");
            this._disconnectSinkAtIndex(validatedIndex);
        } else {
            // Disconnect all outputs
        for (let i = 0; i < this._nexts.length; i++) {
                if (this._nexts[i]) {
                    this._disconnectSinkAtIndex(i);
                }
            }
        }
    }

    private _disconnectSinkAtIndex(index: number): void {
        const next = this._nexts[index];
            if (next) {
                if (next instanceof MergeOp) {
                    // Find our connection in MergeOp's prevs array and remove it
                const mergeIndex = next._prevs.indexOf(this);
                if (mergeIndex >= 0) {
                    next._prevs[mergeIndex] = null as unknown as GraphNode;
                    }
                } else {
                    next.prev = null;
                }
                // Clear the reference with a cast to avoid type error
            this._nexts[index] = null as unknown as GraphNode;
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

        // For reduction operations, output shape matches the first input shape
        // as all inputs must have the same shape for proper reduction
        const referenceShape = [...this._inShapes[0]];
        
        // Validate all shapes have the same dimensions
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

        // Use reduction operation from parameters (e.g., 'add', 'mul')
        const op = this._opType.toLowerCase();
        
        // For most associative operations, we start with the first input and fold in the rest
        if (inputs.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`; // Just return the single input
        }
        
        // Apply the operation sequentially (order matters for non-commutative operations)
        let code = inputs[0];
        for (let i = 1; i < inputs.length; i++) {
            code = `torch.${op}(${code}, ${inputs[i]})`;
        }
        
        return `${inputs[0]} = ${code}`;
    }

    // Override the connectSource method to not require explicit index
    connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        // ReduceOp allows automatic indexing for convenience
        if (indexSelf === undefined) {
            // Find the first available slot
            indexSelf = this._prevs.findIndex(p => !p);
            if (indexSelf === -1) {
                // If no empty slots, add at the end
                indexSelf = this._prevs.length;
                // Expand inShapes array if needed
                if (indexSelf >= this._inShapes.length) {
                    // Clone the first shape to expand
                    this._inShapes.push([...this._inShapes[0]]);
                }
            }
        }
        
        // Call the parent implementation with the determined index
        super.connectSource(prev, indexSelf, indexPrev);
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

        // Use commutative operation from parameters (e.g., 'max', 'min')
        const op = this._opType.toLowerCase();
        
        if (inputs.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`; // Just return the single input
        }
        
        // For commutative operations, we can use a more concise approach
        return `${inputs[0]} = torch.${op}(torch.stack([${inputs.join(", ")}], 0), 0)[0]`;
    }
}

export class MapOp extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        opType: string,
        params: { outputShapes: number[][] }
    ) {
        super(id, inShape, target, opType, params);
    }

    protected computeOutShapes(): number[][] {
        // MapOp has predefined output shapes
        return this._params.outputShapes;
    }

    to_torch_functional(input: string): string[] {
        // Apply the mapping operation to each output
        const operation = this._opType.toLowerCase();
        const outputs = this._outShapes.map((_, i) => `out${i}`);
        
        // Generate different code based on the mapping operation
        if (operation === "split") {
            const splitSizes = this._outShapes.map(shape => shape[this._params.dim]);
            return [`${outputs.join(", ")} = torch.split(${input}, [${splitSizes.join(", ")}], dim=${this._params.dim})`];
        } else if (operation === "chunk") {
            return [`${outputs.join(", ")} = torch.chunk(${input}, ${this._outShapes.length}, dim=${this._params.dim})`];
        } else {
            // Default mapping behavior
            return outputs.map((out, i) => `${out} = torch.index_select(${input}, ${this._params.dim}, torch.tensor(${i}))`);
        }
    }

    // Override the connectSink method to not require explicit index
    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        // MapOp allows automatic indexing for BranchOp outputs
        if (indexSelf === undefined) {
            // Find the first available slot
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                // If no empty slots, we cannot automatically connect
                throw new Error("Cannot automatically determine output index for MapOp. Please specify indexSelf.");
            }
        }
        
        // Call the parent implementation with the determined index
        super.connectSink(next, indexSelf, indexNext);
    }
}

export class Broadcast extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        params: { copies: number }
    ) {
        super(id, inShape, target, "Broadcast", params);
    }

    protected computeOutShapes(): number[][] {
        // All output shapes are identical to input shape
        return Array(this._params.copies).fill(0).map(() => [...this._inShape]);
    }

    to_torch_functional(input: string): string[] {
        // For broadcast, simply assign the same input to all outputs
        return Array(this._params.copies).fill(0).map((_, i) => `out${i} = ${input}`);
    }

    // Override connectSink to be more flexible
    connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        // For Broadcast, if indexSelf is undefined, we can dynamically add new outputs
        if (indexSelf === undefined) {
            // Use the next available index or add a new one
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                // If all slots are filled, add a new output
                indexSelf = this._nexts.length;
                this._params.copies += 1;
                this._outShapes = this.computeOutShapes();
            }
        } else if (indexSelf >= this._params.copies) {
            // If the specified index is beyond the current size, expand
            this._params.copies = indexSelf + 1;
            this._outShapes = this.computeOutShapes();
        }
        
        // Call the parent implementation with the determined index
        super.connectSink(next, indexSelf, indexNext);
    }
}

export class Graph {
  // Efficient storage for all nodes by their ID
  private _nodes: Map<string, GraphNode> = new Map();
  
  // Keep track of source and sink nodes
  private _sources: Map<string, GraphNode> = new Map();
  private _sinks: Map<string, GraphNode> = new Map();

  constructor() {}

  /**
   * Add a node to the graph
   * @param node The node to add
   * @returns The node that was added
   */
  addNode(node: GraphNode): GraphNode {
    if (this._nodes.has(node.id)) {
      throw new Error(`Node with ID ${node.id} already exists in the graph`);
    }
    this._nodes.set(node.id, node);
    // Initially, all nodes are both sources and sinks until connected
    this._sources.set(node.id, node);
    this._sinks.set(node.id, node);
    return node;
  }

  /**
   * Get a node by its ID
   * @param nodeId The ID of the node to get
   * @returns The node with the given ID
   */
  getNode(nodeId: string): GraphNode {
    const node = this._nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node with ID ${nodeId} no longer exists in the graph`);
    }
    return node;
  }

  /**
   * Connect a node to its predecessor
   * @param nodeId The ID of the node to connect
   * @param prevId The ID of the predecessor node
   * @param indexSelf Optional index for multi-input nodes
   * @param indexPrev Optional index for multi-output nodes
   */
  connectNodeToPrev(nodeId: string, prevId: string, indexSelf?: number, indexPrev?: number): void {
    const node = this.getNode(nodeId);
    const prev = this.getNode(prevId);

    node.connectSource(prev, indexSelf, indexPrev);
    
    // Update sources and sinks
    this._sources.delete(nodeId); // No longer a source
    this._sinks.delete(prevId);   // No longer a sink
    
    // Update sink status for the connected nodes
    this._refreshNodeSinkSourceStatus(node);
    this._refreshNodeSinkSourceStatus(prev);
  }

  /**
   * Connect a node to its successor
   * @param nodeId The ID of the node to connect
   * @param nextId The ID of the successor node
   * @param indexSelf Optional index for multi-output nodes
   * @param indexNext Optional index for multi-input nodes
   */
  connectNodeToNext(nodeId: string, nextId: string, indexSelf?: number, indexNext?: number): void {
    const node = this.getNode(nodeId);
    const next = this.getNode(nextId);

    node.connectSink(next, indexSelf, indexNext);
    
    // Update sources and sinks
    this._sources.delete(nextId); // No longer a source
    this._sinks.delete(nodeId);   // No longer a sink
    
    // Update source/sink status for the connected nodes
    this._refreshNodeSinkSourceStatus(node);
    this._refreshNodeSinkSourceStatus(next);
  }

  /**
   * Disconnect a node from its predecessor(s)
   * @param nodeId The ID of the node to disconnect
   * @param indexSelf Optional index for multi-input nodes
   */
  disconnectNodeFromPrev(nodeId: string, indexSelf?: number): void { 
    const node = this.getNode(nodeId);
    
    // Store prev reference(s) before disconnecting
    let prevs: GraphNode[] = [];
    
    if (node instanceof MergeOp) {
      if (indexSelf !== undefined) {
        // If we're disconnecting a specific input, only track that one
        const prev = node._prevs[indexSelf];
        if (prev) prevs = [prev];
      } else {
        // Otherwise, track all inputs
        prevs = node._prevs.filter(Boolean);
      }
    } else if (node.prev) {
      prevs = [node.prev];
    }
    
    node.disconnectSource(indexSelf);
    
    // Update node status
    this._refreshNodeSinkSourceStatus(node);
    
    // Update prev node statuses
    prevs.forEach(prev => {
      if (prev) {
        this._refreshNodeSinkSourceStatus(prev);
      }
    });
  }

  /**
   * Disconnect a node from its successor(s)
   * @param nodeId The ID of the node to disconnect
   * @param indexSelf Optional index for multi-output nodes
   */
  disconnectNodeFromNext(nodeId: string, indexSelf?: number): void {
    const node = this.getNode(nodeId);
    
    // Store next reference(s) before disconnecting
    let nexts: GraphNode[] = [];
    
    if (node instanceof BranchOp) {
      if (indexSelf !== undefined) {
        // If we're disconnecting a specific output, only track that one
        const next = node.nexts[indexSelf];
        if (next) nexts = [next];
      } else {
        // Otherwise, track all outputs
        nexts = node.nexts.filter(Boolean);
      }
    } else if (node.next) {
      nexts = [node.next];
    }
    
    node.disconnectSink(indexSelf);
    
    // Update node status
    this._refreshNodeSinkSourceStatus(node);
    
    // Update next node statuses
    nexts.forEach(next => {
      if (next) {
        this._refreshNodeSinkSourceStatus(next);
      }
    });
  }

  /**
   * Delete a node from the graph
   * @param nodeId The ID of the node to delete
   */
  deleteNode(nodeId: string): void {
    const node = this.getNode(nodeId);
    
    // Disconnect from all connections first
    node.disconnectSource();
    node.disconnectSink();
    
    // Remove from all collections
    this._nodes.delete(nodeId);
    this._sources.delete(nodeId);
    this._sinks.delete(nodeId);
  }

  /**
   * Swap a node with a new node
   * @param oldNodeId The ID of the node to replace
   * @param newNode The new node to replace it with
   * @returns The new node
   */
  swapNode(oldNodeId: string, newNode: GraphNode): GraphNode {
    const oldNode = this.getNode(oldNodeId);
    
    // Store connections
    const prevNode = oldNode.prev;
    let nextNodes: GraphNode[] = [];
    
    if (oldNode instanceof BranchOp) {
      nextNodes = oldNode.nexts.filter(Boolean);
    } else if (oldNode.next) {
      nextNodes = [oldNode.next];
    }
    
    // Remove old node
    this.deleteNode(oldNodeId);
    
    // Add new node
    this.addNode(newNode);
    
    // Recreate connections
    if (prevNode) {
      newNode.connectSource(prevNode);
    }
    
    for (const next of nextNodes) {
      if (next) {
        newNode.connectSink(next);
      }
    }
    
    return newNode;
  }

  /**
   * Generate PyTorch code for the current graph
   * @returns PyTorch code as a string
   */
  to_torch(): string {
    // Validate the graph first
    if (!this.validate_torch()) {
      throw new Error("Cannot generate PyTorch code for invalid graph");
    }
    
    const code: string[] = [];
    code.push("import torch");
    code.push("import torch.nn as nn");
    code.push("import torch.nn.functional as F");
    code.push("");
    
    // Function header
    code.push("def model(input_tensors):");
    
    // Track variables for each node
    const varNames = new Map<string, string>();
    
    // Process sources first
    const sourceIds = Array.from(this._sources.keys());
    for (let i = 0; i < sourceIds.length; i++) {
      const source = this._sources.get(sourceIds[i])!;
      const varName = `x${i}`;
      varNames.set(source.id, varName);
      
      code.push(`    # Source: ${source.id}`);
      code.push(`    ${varName} = input_tensors[${i}]`);
    }
    
    code.push("");
    
    // For visited tracking
    const visited = new Set<string>();
    const outVars: string[] = [];
    
    // Process nodes in topological order
    const processNodeInOrder = (nodeId: string): string => {
      if (visited.has(nodeId)) {
        return varNames.get(nodeId)!;
      }
      
      const node = this._nodes.get(nodeId)!;
      visited.add(nodeId);
      
      // For single-input nodes
      if (!(node instanceof MergeOp)) {
        let inputVar = "";
        
        if (node.prev) {
          inputVar = processNodeInOrder(node.prev.id);
        } else {
          // Source already processed
          inputVar = varNames.get(nodeId)!;
          return inputVar;
        }
        
        const outputVar = `x_${nodeId.substring(0, 4)}`;
        varNames.set(nodeId, outputVar);
        
        code.push(`    # Node: ${node.id} (${node.constructor.name})`);
        
        if (node instanceof Tensor) {
          code.push(`    ${outputVar} = ${inputVar}  # Tensor pass-through`);
        } else if (node instanceof Op) {
          const opCode = node.to_torch_functional(inputVar);
          code.push(`    ${opCode.replace(inputVar + " = ", outputVar + " = ")}`);
        } else if (node instanceof BranchOp) {
          // For BranchOp, need to create multiple outputs
          if (node instanceof Split) {
            const outputNames = node.outShape.map((_, idx) => `${outputVar}_${idx}`);
            code.push(`    ${outputNames.join(", ")} = torch.split(${inputVar}, [${node.params.sections.join(", ")}], dim=${node.params.dim})`);
            
            // Store all output vars for potential use
            outputNames.forEach((name, idx) => {
              varNames.set(`${nodeId}_out${idx}`, name);
            });
            
            // Use the first output as the default if needed
            varNames.set(nodeId, outputNames[0]);
          }
        }
        
        return outputVar;
      } else {
        // For MergeOp (multi-input)
        const inputs: string[] = [];
        
        // Since _prevs is public in MergeOp, we can use it directly
        for (let i = 0; i < (node as MergeOp)._prevs.length; i++) {
          const prev = (node as MergeOp)._prevs[i];
          if (prev) {
            inputs.push(processNodeInOrder(prev.id));
          }
        }
        
        const outputVar = `x_${nodeId.substring(0, 4)}`;
        varNames.set(nodeId, outputVar);
        
        code.push(`    # Node: ${node.id} (${node.constructor.name})`);
        
        if (node instanceof Concat) {
          code.push(`    ${outputVar} = torch.cat([${inputs.join(", ")}], dim=${node.params.dim})`);
        } else {
          // Other merge ops...
          const mergeOpCode = node.to_torch_functional(inputs);
          code.push(`    ${mergeOpCode.replace(inputs[0] + " = ", outputVar + " = ")}`);
        }
        
        return outputVar;
      }
    };
    
    // Process sink nodes to get outputs
    for (const sinkId of this._sinks.keys()) {
      const outVar = processNodeInOrder(sinkId);
      outVars.push(outVar);
    }
    
    // Return outputs
    code.push("");
    code.push("    # Return outputs");
    
    if (outVars.length === 1) {
      code.push(`    return ${outVars[0]}`);
    } else {
      code.push(`    return [${outVars.join(", ")}]`);
    }
    
    return code.join("\n");
  }

  /**
   * Validate the graph for PyTorch code generation
   * - No loops
   * - All sinks reachable from sources
   * @returns true if valid, false if invalid
   */
  validate_torch(): boolean {
    // If graph is empty, it's valid by default
    if (this._nodes.size === 0) {
      return true;
    }
    
    // If no sources or no sinks, it's invalid
    if (this._sources.size === 0 || this._sinks.size === 0) {
      return false;
    }
    
    // Check for loops and reachability
    const visited = new Set<string>();
    const visiting = new Set<string>();
    
    // DFS with cycle detection
    const dfs = (nodeId: string): boolean => {
      if (visited.has(nodeId)) {
        return true; // Already fully visited, no cycles
      }
      
      if (visiting.has(nodeId)) {
        return false; // Cycle detected
      }
      
      visiting.add(nodeId);
      
      const node = this._nodes.get(nodeId)!;
      
      // Check next nodes
      if (node instanceof BranchOp) {
        for (const next of node.nexts) {
          if (next && !dfs(next.id)) {
            return false; // Cycle detected in branch
          }
        }
      } else if (node.next) {
        if (!dfs(node.next.id)) {
          return false; // Cycle detected
        }
      }
      
      visiting.delete(nodeId);
      visited.add(nodeId);
      return true;
    };
    
    // Run DFS from all sources
    for (const sourceId of this._sources.keys()) {
      if (!dfs(sourceId)) {
        return false; // Cycle detected
      }
    }
    
    // Check if all sinks are reachable from sources
    for (const sinkId of this._sinks.keys()) {
      if (!visited.has(sinkId)) {
        return false; // Sink not reachable
      }
    }
    
    return true; // No cycles and all sinks reachable
  }

  /**
   * Get all nodes in the graph
   * @returns Array of all nodes
   */
  getNodes(): GraphNode[] {
    return Array.from(this._nodes.values());
  }

  /**
   * Get all source nodes in the graph
   * @returns Array of source nodes
   */
  getSources(): GraphNode[] {
    return Array.from(this._sources.values());
  }

  /**
   * Get all sink nodes in the graph
   * @returns Array of sink nodes
   */
  getSinks(): GraphNode[] {
    return Array.from(this._sinks.values());
  }

  /**
   * Update the source/sink status of a node
   * @param node The node to update
   */
  private _refreshNodeSinkSourceStatus(node: GraphNode): void {
    const id = node.id;
    
    // Check if node is a source (no inputs)
    if (node instanceof MergeOp) {
      // MergeOp is a source if all inputs are null
      const hasInputs = node._prevs.some(Boolean);
      if (!hasInputs) {
        this._sources.set(id, node);
      } else {
        this._sources.delete(id);
      }
    } else {
      // Other nodes are sources if prev is null
      if (node.prev === null) {
        this._sources.set(id, node);
      } else {
        this._sources.delete(id);
      }
    }
    
    // Check if node is a sink (no outputs)
    if (node instanceof BranchOp) {
      // BranchOp is a sink if it has no nexts or all nexts are null
      const hasOutputs = node.nexts.some(Boolean);
      if (!hasOutputs) {
        this._sinks.set(id, node);
      } else {
        this._sinks.delete(id);
      }
    } else {
      // Other nodes are sinks if next is null
      if (node.next === null) {
        this._sinks.set(id, node);
      } else {
        this._sinks.delete(id);
      }
    }
  }
}






