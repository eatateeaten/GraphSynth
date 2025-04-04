import { GraphNode } from './graph_node';

export abstract class BranchOp extends GraphNode {
    protected readonly _opType: string;
    protected readonly _numberOfBranches: number;

    constructor(
        id: string,
        opType: string,
        numberOfBranches: number,
        params: Record<string, any>,
    ) {
        super(id, params);
        
        this._inShapes = [null];
        this._outShapes = Array(numberOfBranches).fill(null);
        this._prevs = [null];
        this._nexts = Array(numberOfBranches).fill(null);

        this._opType = opType;
        this._numberOfBranches = numberOfBranches;
    }

    protected abstract computeOutShape(): number[][];
    abstract to_torch_functional(inputs: string[], outputs: string[]): string;

    // Getters and setters 
    get opType(): string { return this._opType; }
    set params(params: Record<string, any>) {
        // Make a deep copy to avoid modifying the original object
        (this._params) = { ...params };
        
        // Recalculate output shapes
        try {
            this._outShapes = this.computeOutShape();
        } catch (err: any) {
            // If shape inference fails, we keep the existing output shapes
            throw new Error(`Failed to update output shapes after params change: ${err.message}`);
        }
    }

    addPrev(prev: GraphNode, prevOutShape: number[]): void {
        if (this._prevs[0] !== null) {
            throw new Error("Branch already has a source connection");
        }

        // Get the output shape from the source node        
        // Set inShape and compute outShape
        this._inShapes[0] = [...prevOutShape];

        try {
            this._outShapes = this.computeOutShape();
        } catch (err: any) {
            // Reset inShape if shape inference fails
            this._inShapes[0] = null;
            throw err;
        }

        // Set our prev reference
        this._prevs[0] = prev; 
    }

    addNext(next: GraphNode, indexSelf: number): void {
        if (this._nexts[indexSelf] !== null) {
            throw new Error(`BranchOp already has a connection at output ${indexSelf}`);
        }
        this._nexts[indexSelf] = next; 
    }

    deletePrev(): void {
        this._prevs = [null];
        this._inShapes = [null];
        this._outShapes = [null];
    }

    deleteNext(indexSelf?: number): void {
        if (indexSelf === undefined) {
            // Clear all next connections
            this._nexts.fill(null);
            return;
        }

        // Validate index
        const validatedIndex = GraphNode.checkIndexInBound(indexSelf, this._numberOfBranches, "BranchOp.deleteNext");
        
        // Clear the specific connection
        this._nexts[validatedIndex] = null;
    }
}


export class Split extends BranchOp {
    private _dim: number;
    private _sections: number[];

    constructor(
        id: string,
        dim: number,
        sections: number[],
        params: Record<string, any>
    ) {
        super(id, "Split", sections.length, params);
        this._dim = dim;
        this._sections = sections;
    }

    /* todo: reset dim and sections when params are set */
    /* todo: setParams should probably be called from the constructor
     * and should be mandatory for every non-abstract device? */

    protected computeOutShape(): number[][] {
        const { dim, sections } = this._params;
        const outShapes: number[][] = [];
        
        if (this._inShapes[0] === null) {
            throw new Error("Input shape must be defined to compute output shapes");
        }

        let start = 0;
        for (const size of sections) {
            const outShape = [...this._inShapes[0]];
            // Each section starts at 'start' and has length 'size'
            outShape[dim] = size;
            outShapes.push(outShape);
            start += size;
        }

        // Verify total size matches input shape
        if (start !== this._inShapes[0][dim]) {
            throw new Error(`Total split size ${start} does not match input dimension ${this._inShapes[0][dim]}`);
        }
        return outShapes;
    }

    to_torch_functional(inputs: string[], outputs: string[]): string {
        const dim = this._dim;
        const sections = this._sections;

        if (sections.length === 1) {
            // Only one section means just a simple copy
            return `${outputs[0]} = ${inputs[0]}`;
        }

        // For multi-output splits with unpacking
        // torch.split returns a tuple that needs to be assigned to the output variables
        return `${outputs.join(', ')} = torch.split(${inputs[0]}, sections=${JSON.stringify(sections).replace('[', '(').replace(']', ')')}, dim=${dim})`;
    }
} 

/**
 * Copy operation creates multiple identical outputs from a single input.
 * Each output has the same shape as the input.
 */
export class Copy extends BranchOp {
    constructor(
        id: string,
        copies: number,
        params: Record<string, any>,
    ) {
        super(id, "Copy", copies, params);
    }

    protected computeOutShape(): number[][] {
        const { copies } = this._params;
        const outShapes: number[][] = []; 
        
        if (this._inShapes[0] === null) {
            throw new Error("Input shape must be defined to compute output shapes");
        }
    
        for (let i = 0; i < copies; i++) {
            outShapes.push([...this._inShapes[0]]);
        } 

        return outShapes;
    }

    to_torch_functional(inputs: string[], outputs: string[]): string {
        // Handle multiple outputs separately to make each assignment clear in the output
        if (outputs.length <= 1) {
            return `${outputs[0]} = ${inputs[0]}`;
        }
        
        // Generate separate assignment statements for each output
        // This ensures each output gets its own line and is treated as a separate variable
        return outputs.map(output => `${output} = ${inputs[0]}`).join('\n');
    }
} 
