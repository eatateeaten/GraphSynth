/**
 * Class Diagram:
 * 
 * GraphNode (base)
 *    ↑
 * MergeOp (abstract)
 *    ↑
 * ReduceOp (base)
 *    ↑
 * ├── PointwiseReduce
 * └── Concat
 * 
 * Key relationships:
 * - ReduceOp extends MergeOp
 * - PointwiseReduce and Concat extend ReduceOp
 * - All classes inherit from GraphNode
 * 
 * Key methods:
 * - computeOutShape(): abstract in MergeOp, implemented in ReduceOp and its subclasses
 * - to_torch_functional(): abstract in MergeOp, implemented in ReduceOp and its subclasses
 * - addPrev(): overridden in ReduceOp to handle dynamic input shapes
 */

import { MergeOp } from './merge_op';
import { getElementwiseOpCode } from './torch_nn_module_op';
import { GraphNode } from './graph_node';

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

        const referenceShape = [...this._inShapes[0]];
        
        for (let i = 1; i < this._inShapes.length; i++) {
            const shape = this._inShapes[i];
            if (!GraphNode.shapeMatch(referenceShape, shape)) {
                throw new Error(`For reduction operations, all input shapes must match. Shape at index ${i} [${shape}] doesn't match reference shape [${referenceShape}]`);
            }
        }

        return referenceShape;
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length < 1) {
            throw new Error("ReduceOp requires at least 1 input");
        }

        const op = this._opType.toLowerCase();
        
        if (inputs.length === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        let code = inputs[0];
        for (let i = 1; i < inputs.length; i++) {
            code = `torch.${op}(${code}, ${inputs[i]})`;
        }
        
        return `${inputs[0]} = ${code}`;
    }

    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void {
        if (indexSelf === undefined) {
            indexSelf = this._prevs.findIndex(p => !p);
            if (indexSelf === -1) {
                indexSelf = this._prevs.length;
                if (indexSelf >= this._inShapes.length) {
                    this._inShapes.push([...this._inShapes[0]]);
                }
            }
        }
        
        super.addPrev(prev, indexSelf, indexPrev);
    }
}

export class PointwiseReduce extends ReduceOp {
    constructor(id: string, inShapes: number[][], target: string, opType: string) {
        super(id, inShapes, target, opType, {});
    }

    protected computeOutShape(): number[] {
        if (this._inShapes.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 input tensors");
        }

        let resultShape = [...this._inShapes[0]];

        for (let i = 1; i < this._inShapes.length; i++) {
            const currentShape = this._inShapes[i];
            
            if (currentShape.length !== resultShape.length) {
                throw new Error(`Input shapes must have the same rank. Shape at index ${i} has rank ${currentShape.length}, expected ${resultShape.length}`);
            }
            
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

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        if (inputs.length < 2) {
            throw new Error("PointwiseReduce requires at least 2 inputs");
        }
        const opCode = getElementwiseOpCode(this._opType);
        
        const result = inputs.reduce((acc, curr) => 
            acc ? `${opCode}(${acc}, ${curr})` : curr
        );
        
        return `${inputs[0]} = ${result}`;
    }
}

export class Concat extends ReduceOp {
    constructor(id: string, inShapes: number[][], target: string, params: { dim: number }) {
        super(id, inShapes, target, "Concat", params);
    }

    protected computeOutShape(): number[] {
        const dim = this._params.dim;
        if (dim < 0 || dim >= this._inShapes[0].length) {
            throw new Error(`Invalid concatenation dimension ${dim} for input shape of length ${this._inShapes[0].length}`);
        }

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

        const outShape = [...referenceShape];
        outShape[dim] = this._inShapes.reduce((sum, shape) => sum + shape[dim], 0);
        return outShape;
    }

    to_torch_functional(inputs: string[], outputs?: string[]): string {
        return `${inputs[0]} = torch.cat([${inputs.join(', ')}], dim=${this._params.dim})`;
    }
}