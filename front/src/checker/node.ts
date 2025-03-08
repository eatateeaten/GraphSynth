/**
 * TypeScript port of the Python shape-checking graph.
 * 
 * Key implementation notes:
 * - Shape = number[] (equivalent to Python List[int])
 * - Uses TypeScript classes instead of Python ABC
 * - Each node must implement:
 *     1. compute_out_shape(in_shape: Shape): Shape
 *     2. validate_params(): void
 * 
 * The main difference is how output shapes work:
 * - out_shape is only recomputed when inputs or params change
 * - we raise an error if the shape doesn't match the next node's input shape
 *      UI is expected to disconnect the nodes in this case
 * - compute_out_shape() is pure - just takes in_shape & params and returns result
 * - update_out_shape() handles the caching and error handling
 * - so in that sense I assumed the left to right building thingy
 */

import { Shape } from './shape';

// Error types for different failure cases
export class ValidationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'ValidationError';
    }
}
export class InputError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'InputError';
    }
}
export class OutputError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'OutputError';
    }
}

export interface NodeParams {
    [key: string]: any;
}

export type ParamFieldMetadata = {
    label: string;
    description: string;
    type: 'shape' | 'number' | 'option';
    allowNegativeOne?: boolean;
    options?: string[];
    default?: number | string | Shape;
}

export type NodeMetadata<T extends NodeParams = NodeParams> = {
    label: string;
    description: string;
    category: 'basic' | 'convolution' | 'pooling' | 'activation';
    paramFields: {
        [K in keyof T]: ParamFieldMetadata;
    };
}

export abstract class CheckerNode<T extends NodeParams = NodeParams> {
    static readonly type: string;
    static readonly description: string = '';
    
    // Add a getter to access the static type from instances
    get type(): string {
        return (this.constructor as typeof CheckerNode).type;
    }

    static getMeta(..._args: any[]): NodeMetadata {
        throw new Error('getMeta() not implemented');
    }

    /**
     * Static validation that can be used before node creation.
     * This is treated as an abstract static method, but TypeScript doesn't support 
     * abstract static methods, so we throw an error instead.
     * All derived classes MUST implement this method.
     */
    static validateParams(_params: NodeParams): string | null {
        throw new Error(`validateParams() not implemented`);
    }

    in_shape: Shape | null = null;
    out_shape: Shape | null = null;
    in_node: CheckerNode<any> | null = null;
    out_node: CheckerNode<any> | null = null;
    params: T;

    constructor(params: T) {
        const error = this.validateParams(params);
        if (error) throw new Error(error);
        
        this.params = params;
        this.updateOutShape();
    }

    abstract computeOutShape(in_shape: Shape): Shape;

    /**
     * Instance method that delegates to static validation.
     * This is implemented in the base class and should NOT be overridden.
     */
    validateParams(params: T = this.params): string | null {
        return (this.constructor as any).validateParams(params);
    }

    protected updateOutShape(): void {
        if (this.in_shape === null) {
            this.out_shape = null;
            return;
        }

        try {
            this.out_shape = this.computeOutShape(this.in_shape);
        } catch (e) {
            throw new OutputError(`Output shape computation failed: ${e instanceof Error ? e.message : String(e)}`);
        }

        if (this.out_node && !Shape.equals(this.out_shape, this.out_node.in_shape)) {
            throw new OutputError(
                `Output shape mismatch with subsequent input shape: ${this.out_shape} vs ${this.out_node.in_shape}`
            );
        }
    }

    setParams(params: T): void {
        const error = this.validateParams(params);
        if (error) throw new ValidationError(`Parameter validation failed: ${error}`);
        
        this.params = params;
        this.updateOutShape();
    }

    connectTo(target: CheckerNode<any>): void {
        if (target.in_shape !== null) {
            if (!Shape.equals(this.out_shape, target.in_shape)) {
                throw new OutputError(
                    `Input shape mismatch: cannot connect output shape ${this.out_shape} to input shape ${target.in_shape}`
                );
            }
        } else {
            target.setInShape(this.out_shape);
        }
        this.out_node = target;
        target.in_node = this;
    }

    setInShape(shape: Shape | null): void {
        // Try computing output shape first before committing to the input shape
        try {
            const newOutShape = shape ? this.computeOutShape(shape) : null;
            // Only set shapes if computation succeeds
            this.in_shape = shape;
            this.out_shape = newOutShape;
        } catch (e) {
            if (e instanceof OutputError) {
                throw new InputError(e.message);
            }
            throw e;
        }
    }
}
