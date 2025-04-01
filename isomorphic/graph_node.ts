/**
 * Base types for the graph implementation.
 * This file is used to prevent circular imports.
 */
export abstract class GraphNode {
    protected readonly _id: string;
    protected readonly _target: string;
    constructor(id: string, target: string) {
        // Validate UUID format
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        if (!uuidRegex.test(id)) {
            throw new Error(`Invalid UUID format: ${id}`);
        }
        this._id = id;
        this._target = target;
    }

    get id(): string { return this._id; }
    get target(): string { return this._target; }

    abstract get prev(): GraphNode | null;
    abstract set prev(node: GraphNode | null);
    abstract get next(): GraphNode | null;
    abstract set next(node: GraphNode | null);

    // Abstract shape and parameter accessors
    abstract get inShape(): number[] | number[][] | null;
    abstract get outShape(): number[] | number[][] | null;
    abstract get params(): Record<string, any>;
    abstract set params(params: Record<string, any>);

    abstract addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void;
    abstract addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract deletePrev(indexSelf?: number): void;
    abstract deleteNext(indexSelf?: number): void;
    abstract to_torch_functional(inputs: string[], outputs?: string[]): string;

    static checkIndexInBound(index: number, length: number, context: string): number {
        if (index < 0 || index >= length) {throw new Error(`${context}: Index ${index} is out of bounds for length ${length}`);}
        return index;
    }

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
        // Check if the node is of a type that infers its input shape
        const className = node.constructor.name;
        return className.includes('MergeOp') || 
               className.includes('BranchOp') || 
               className === 'Op' ||
               (!className.includes('Tensor') && !className.includes('Module'));
    }


    /**
     * Checks if the node can have multiple static inputs
     * @param node The graph node to check
     * @returns true if the node can have multiple static inputs
     */
    static multipleStaticInputs(node: GraphNode): boolean {
        const className = node.constructor.name;
        return className.includes('Module');
    }

    /**
     * Checks if the node can have multiple inputs
     * @param node The graph node to check
     * @returns true if the node can have multiple inputs
     */
    static multipleOutputs(node: GraphNode): boolean {
        const className = node.constructor.name;
        return className.includes('BranchOp') || className.includes('Module');
    }
    
    /**
     * Checks if the node can have multiple inputs
     * @param node The graph node to check
     * @returns true if the node can have multiple inputs
     * By default we won't be supporting arbitrary length of input 
     */
    static multipleInputs(node: GraphNode): boolean {
        const className = node.constructor.name;
        return className.includes('MergeOp') || className.includes('Module');
    }
    
    /**
     * Checks if the node can have multiple static outputs
     * @param node The graph node to check
     * @returns true if the node can have multiple static outputs
     */
    static multipleStaticOutputs(node: GraphNode): boolean {
        const className = node.constructor.name;
        return className.includes('Module') || className.includes('Split') || className.includes('Copy');
    }

    /**
     * Checks if the node can only have a single input
     * @param node The graph node to check
     * @returns true if the node can only have a single input
     */
    static singleInput(node: GraphNode): boolean {
        // Direct class name check for maximum clarity and reliability
        const className = node.constructor.name;
        
        // Multi-input node types (explicit listing)
        const multiInputTypes = [
            'MergeOp',
            'Concat',
            'PointwiseReduce',
            'PointwiseOp',
            'DotOp',
            'CrossOp'
        ];
        
        // Check if class name matches any of the multi-input types
        if (multiInputTypes.some(type => className === type || className.includes(`${type}Module`))) {
            return false;
        }
        
        // Any module-type node has multiple inputs
        if (className.includes('Module')) {
            return false;
        }
        
        // All other nodes have single inputs
        return true;
    }

    /**
     * Checks if the node can only have a single output
     * @param node The graph node to check
     * @returns true if the node can only have a single output
     */
    static singleOutput(node: GraphNode): boolean {
        // Direct class name check for maximum clarity and reliability
        const className = node.constructor.name;
        
        // Multi-output node types (explicit listing)
        const multiOutputTypes = [
            'BranchOp',
            'Split',
            'Copy'
        ];
        
        // Check if class name matches any of the multi-output types
        if (multiOutputTypes.some(type => className === type || className.endsWith(type))) {
            return false;
        }
        
        // Any module-type node has multiple outputs
        if (className.includes('Module')) {
            return false;
        }
        
        // All other nodes have single outputs
        return true;
    }

    /**
     * Checks if the node has any input connections
     * @param node The graph node to check
     * @returns true if the node has any inputs connected
     */
    static hasInputs(node: GraphNode): boolean {
        if (node.constructor.name.includes('MergeOp')) {
            // @ts-ignore: Accessing protected property
            return !node._prevs.every(p => !p);
        } else {
            return node.prev !== null;
        }
    }

    /**
     * Checks if the node has any output connections
     * @param node The graph node to check
     * @returns true if the node has any outputs connected
     */
    static hasOutputs(node: GraphNode): boolean {
        if (node.constructor.name.includes('BranchOp')) {
            // @ts-ignore: Accessing protected property
            return !node._nexts.every(n => !n);
        } else {
            return node.next !== null;
        }
    }
} 