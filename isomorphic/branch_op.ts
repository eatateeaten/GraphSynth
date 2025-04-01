import { GraphNode } from './graph_node';

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
    abstract to_torch_functional(inputs: string[], outputs: string[]): string;

    // Getters and setters 
    get inShape(): number[] { return this._inShape; }
    get outShape(): number[][] { return this._outShapes; }
    get prev(): GraphNode | null { return this._prev; }
    set prev(node: GraphNode | null) { this._prev = node; }
    get next(): GraphNode | null { return null; }
    set next(node: GraphNode | null) { /* Do nothing */ }
    get nexts(): GraphNode[] { return this._nexts; }
    get opType(): string { return this._opType; }
    get params(): Record<string, any> { return { ...this._params }; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params as Record<string, any>) = { ...params };
        
        // Recalculate output shapes
        try {
            this._outShapes = this.computeOutShapes();
        } catch (err: any) {
            // If shape inference fails, we keep the existing output shapes
            console.warn(`Failed to update output shapes after params change: ${err.message}`);
        }
    }

    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
        if (this._prev !== null) {
            throw new Error("BranchOp already has a source connection");
        }
        // Just set our prev reference - Graph handles all validation and connections
        this._prev = prev;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        // For BranchOp, indexSelf (output index) is required
        if (indexSelf === undefined) {
            throw new Error("BranchOp.addNext requires an output index");
        }
        
        // Validate and normalize index
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._outShapes.length, "BranchOp.addNext");
        
        // Check if a connection already exists at this output
        if (this._nexts[validatedIndex] !== null && this._nexts[validatedIndex] !== undefined) {
            throw new Error(`BranchOp already has a connection at output ${validatedIndex}`);
        }
        
        // Set the connection
        this._nexts[validatedIndex] = next;
    }

    deletePrev(indexSelf?: number): void {
        // Just clear our reference
        this._prev = null;
    }

    deleteNext(indexSelf?: number): void {
        if (indexSelf === undefined) {
            // Clear all next connections
            this._nexts.fill(null as unknown as GraphNode);
            return;
        }
        
        // Validate index
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._outShapes.length, "BranchOp.deleteNext");
        
        // Clear the specific connection
        this._nexts[validatedIndex] = null as unknown as GraphNode;
    }
}


export class Split extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        params: { dim: number, sections: number[] }
    ) {
        super(id, inShape, target, "Split", params);
    }

    protected computeOutShapes(): number[][] {
        const { dim, sections } = this._params;
        const outShapes: number[][] = [];
        
        for (const size of sections) {
            const outShape = [...this._inShape];
            outShape[dim] = size;
            outShapes.push(outShape);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[], outputs: string[]): string {
        const { dim, sections } = this._params;
        
        if (sections.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        const outputsStr = inputs.map((input, i) => `${input}`).join(', ');
        return `${outputsStr} = torch.split(${inputs[0]}, sections=${JSON.stringify(sections)}, dim=${dim})`;
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }

    // For test compatibility
    get nexts(): GraphNode[] {
        return this._nexts;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                indexSelf = this._nexts.length;
                if (indexSelf >= this._outShapes.length) {
                    // This is not ideal for Split since output shapes depend on sections
                    // But we'll allow it for flexibility
                    const lastShape = [...this._inShape];
                    lastShape[this._params.dim] = 1; // Default to size 1 for new sections
                    this._outShapes.push(lastShape);
                    // Update sections parameter
                    this._params.sections.push(1);
                }
            }
        }
        
        super.addNext(next, indexSelf, indexNext);
    }
} 

/**
 * Copy operation creates multiple identical outputs from a single input.
 * Each output has the same shape as the input.
 */
export class Copy extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        params: { copies: number }
    ) {
        super(id, inShape, target, "Copy", params);
    }

    protected computeOutShapes(): number[][] {
        const { copies } = this._params;
        if (!Number.isInteger(copies) || copies < 1) {
            throw new Error(`Copy operation requires a positive integer number of copies, got: ${copies}`);
        }
        
        // Create 'copies' number of identical output shapes
        const outShapes: number[][] = [];
        for (let i = 0; i < copies; i++) {
            outShapes.push([...this._inShape]);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[], outputs: string[]): string {
        // Simple implementation that just assigns the same input to all outputs
        if (outputs.length === 1) {
            return `${outputs[0]} = ${inputs[0]}`;
        }
        
        return outputs.map(output => `${output} = ${inputs[0]}`).join('\n');
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        if (node !== null) {
            this._nexts.push(node);
        }
    }

    get nexts(): GraphNode[] {
        return this._nexts;
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            // Find the first empty slot or add to the end
            indexSelf = this._nexts.findIndex(n => !n);
            if (indexSelf === -1) {
                indexSelf = this._nexts.length;
                if (indexSelf >= this._outShapes.length) {
                    // If we need more outputs than specified, expand the copies parameter
                    this._params.copies = indexSelf + 1;
                    this._outShapes = this.computeOutShapes();
                }
            }
        }
        
        super.addNext(next, indexSelf, indexNext);
    }
} 

