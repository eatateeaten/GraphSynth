import { GraphNode } from './graph_node';
import { ParamError } from './types';

export class Tensor extends GraphNode {
    protected _variableName: string;

    constructor(id: string, shape: number[], variableName: string) {
        super(id, {});
        this._inShapes = [shape];
        this._outShapes = [shape];
        this._prevs = [null];
        this._nexts = [null];
        this._variableName = variableName;
    }

    /** Validate params and construct if OK */
    static fromParams(id: string, params: Record<string, any>): Tensor {
        if (!params.shape)
            throw new ParamError("Shape is required for Tensor");
        if (!params.variableName) 
            throw new ParamError("Variable name is required for Tensor");
        
        return new Tensor(id, params.shape, params.variableName);
    }

    // Getters and setters
    get variableName(): string { return this._variableName; }
    set variableName(name: string) { this._variableName = name; }
    
    // Implement params accessors
    get params(): Record<string, any> { 
        return {
            shape: [...this._inShapes],
            variableName: this._variableName
        }; 
    }
    set params(params: Record<string, any>) {
        if (params.shape) {
            this._inShapes = [[...params.shape]];
            this._outShapes = [[...params.shape]];
        }
        if ('variableName' in params) {
            this._variableName = params.variableName;
        }
    }

    addPrev(prev: GraphNode): void {
        if (this._prevs[0] !== null) {
            throw new Error("Tensor already has a source connection");
        }

        // Just set our prev reference - Graph handles all validation and connections
        this._prevs[0] = prev;
    }

    addNext(next: GraphNode): void {
        if (this._nexts[0] !== null) {
            throw new Error("Tensor already has a sink connection");
        }
        // Just set our next reference - Graph handles all validation and connections
        this._nexts[0] = next;
    }

    deletePrev(): void {
        // Just clear our reference
        this._prevs = [null];
    }

    deleteNext(): void {
        // Just clear our reference
        this._nexts = [null];
    }

    toTorchModule(): string {
        return "nn.Identity()";
    }

    /**
     * Generates intermediate representation for this tensor.
     * 
     * @returns A string containing the IR representation of this tensor
     */
    toIR(): string {
        const shapeStr = this._outShapes[0] ? `[${this._outShapes[0].join(',')}]` : 'unknown';
        return `Tensor(${this._variableName}) -> ${shapeStr}`;
    }
}
