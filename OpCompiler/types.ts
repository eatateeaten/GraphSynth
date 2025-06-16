/** Types for graph */
const NODE_TYPES = [
    'Tensor', 'Op', 'Split', "Concat", 'Copy',
    'Concat', 'PointwiseReduce', 'PointwiseOp',
    'DotOp', 'CrossOp'
] as const;

export type NodeType = typeof NODE_TYPES[number];

export function isNodeType(str: string): str is NodeType {
    return str in NODE_TYPES;
}

export type Shape = number[];

/** Custom error types */
const createError = (name: string) => class extends Error {
    constructor(message: string) {
        super(message);
        this.name = name;
    }
};

export const ShapeMatchError = createError("ShapeMatchError");
export const ShapeInferenceError = createError("ShapeInferenceError"); 
export const ParamError = createError("ParamError"); 
export const targetError = createError("targetError")

 