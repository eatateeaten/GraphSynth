/** Types for graph */
export type NodeType = 'Tensor' | 'Op' | 'Split' | 'Copy' | 'Concat' | 'PointwiseReduce' | 'DotOp' | 'CrossOp';
export type Shape = number[];
export type TargetType = "Torch" | "JAX";
