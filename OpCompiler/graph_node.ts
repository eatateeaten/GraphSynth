/**
 * Base types for the graph implementation.
 * This file is used to prevent circular imports.
 */
export abstract class GraphNode {
    protected readonly _id: string;
    protected _params: Record<string, any>;

    protected _inShapes: (number[] | null)[] = [];
    protected _outShapes: (number[] | null)[] = [];

    protected _prevs: (GraphNode | null)[] = [];
    protected _nexts: (GraphNode | null)[] = [];

    constructor(id: string, params: Record<string, any>) {
        // Validate UUID format
        // const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        // if (!uuidRegex.test(id)) {
        //     throw new Error(`Invalid UUID format: ${id}`);
        // }
        this._id = id;
        this._params = params;
    }

    get id(): string { return this._id; }

    get prevs(): (GraphNode | null)[] { return this._prevs; }
    get nexts(): (GraphNode | null)[] { return this._nexts; }

    // Abstract shape and parameter accessors
    get inShapes(): (number[] | null)[] { return this._inShapes; }
    get outShapes(): (number[] | null)[] { return this._outShapes; }
    get params(): Record<string, any> { return this._params; }
    set params(params: Record<string, any>) { this._params = params; }

    abstract addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void;
    abstract addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract deletePrev(indexSelf?: number): void;
    abstract deleteNext(indexSelf?: number): void;
    abstract emitTorchModule(inputs: string[], outputs?: string[]): string;
    abstract emitIR(): string;

    static shapeMatch(shape1: number[], shape2: number[]): boolean {
        if (shape1.length !== shape2.length) {
            return false;
        }
        for (let i = 0; i < shape1.length; i++) {
            if (shape1[i] !== shape2[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Checks if the node's input shape is inferred from its inputs
     * @param node The graph node to check
     * @returns true if node is MergeOp, BranchOp, or Op; false if node is Tensor or Module
     */
    static inShapeInferred(node: GraphNode): boolean {
        const className = node.constructor.name;
        
        // List all node types that infer their input shape
        const shapeInferredTypes = [
            'MergeOp',
            'BranchOp',
            'Op',
            'Concat',
            'PointwiseReduce',
            'PointwiseOp',
            'DotOp',
            'CrossOp',
            'Split',
            'Copy'
        ];
        
        // Check if node is a type that infers shape
        return shapeInferredTypes.includes(className) || 
               (!className.includes('Tensor') && !className.includes('Module'));
    }

    /**
     * Checks if the node has any input connections
     * @param node The graph node to check
     * @returns true if the node has any inputs connected
     */
    static hasInputs(node: GraphNode): boolean {
        return node._prevs.some(prev => prev !== null);
    }

    /**
     * Checks if the node has any output connections
     * @param node The graph node to check
     * @returns true if the node has any outputs connected
     */
    static hasOutputs(node: GraphNode): boolean {
        return node._nexts.some(next => next !== null);
    }
}
