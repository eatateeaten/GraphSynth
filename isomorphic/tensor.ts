import { assert } from 'console';
import { GraphNode } from './graph_node';

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

    emit_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length === 0) {
            // This is a source tensor - use variableName if available, otherwise ID
            return this._variableName || this.id;
        }
        const outVar = outputs && outputs.length > 0 ? outputs[0] : inputs[0];
        return `${outVar} = ${inputs[0]}`;
    }
}
