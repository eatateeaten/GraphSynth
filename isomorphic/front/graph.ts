// export class CheckerGraph {
//     private nodes = new Map<string, CheckerNode<any>>();

//     addNode<T extends NodeParams>(id: string, node: CheckerNode<T>): void {
//         this.nodes.set(id, node);
//     }

//     getNode(id: string): CheckerNode<any> | undefined {
//         return this.nodes.get(id);
//     }

//     connect(sourceId: string, targetId: string): void {
//         const source = this.nodes.get(sourceId);
//         const target = this.nodes.get(targetId);
        
//         if (!source || !target) {
//             throw new Error("Node not found");
//         }

//         source.connectTo(target);
//     }

//     deleteNode(id: string): void {
//         const node = this.nodes.get(id);
//         if (!node) return;

//         if (node.in_node) {
//             node.in_node.out_node = null;
//         }
//         if (node.out_node) {
//             node.out_node.in_node = null;
//         }

//         this.nodes.delete(id);
//     }
// }

import { v4 as uuidv4 } from 'uuid';
import { getTorchCode } from './torch_nn_module_op_to_call_map';

export class Op<T> {
    private readonly _id: string;
    protected _inShape: number[];
    protected _outShape: number[];
    private _prev: Op<T> | null;
    private _next: Op<T> | null;
    private _target: string;
    private _opType: string;
    private _params: Record<string, T>;

    constructor(
        inShape: number[],
        outShape: number[],
        target: string,
        opType: string,
        params: Record<string, T>
    ) {
        this._id = uuidv4();
        this._inShape = inShape;
        this._outShape = outShape;
        this._prev = null;
        this._next = null;
        this._target = target;
        this._opType = opType;
        this._params = params;
    }

    to_torch(): string { 
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        } 
        //TODO We need to add the dim_select stuff here otherwise it wouldn't work for some PyTorch nn.module lol lol 
        return getTorchCode(this._opType, this._params);
    }

    to_torch_functional(input: string): string {
        if (this._target !== "torch") {
            throw new Error("Operation is not a PyTorch operation");
        }
        const module = this.to_torch();
        return `${input} = ${module}(${input})`;
    }

    // Getter for id
    get id(): string {
        return this._id;
    }

    // Getters
    get inShape(): number[] {
        return this._inShape;
    }

    get outShape(): number[] {
        return this._outShape;
    }

    get prev(): Op<T> | null {
        return this._prev;
    }

    get next(): Op<T> | null {
        return this._next;
    }

    get target(): string {
        return this._target;
    }

    get opType(): string {
        return this._opType;
    }

    get params(): Record<string, T> {
        return { ...this._params };  // Return a copy to prevent direct mutation
    }

    // Setters
    set prev(op: Op<T> | null) {
        this._prev = op;
    }

    set next(op: Op<T> | null) {
        this._next = op;
    }
}

export class Seq<T> extends Op<T> implements Iterable<Op<T>> {
    private _operations: Op<T>[];

    constructor(initialOp: Op<T>) {
        super(
            initialOp.inShape,
            initialOp.outShape,
            initialOp.target,
            "Seq",
            {}
        );
        this._operations = [initialOp];
    }

    to_torch(): string {
        // Verify all operations are PyTorch operations
        for (const op of this._operations) {
            if (op.target !== "torch") {
                throw new Error("All operations in sequence must be PyTorch operations");
            }
        }

        // Generate sequential module code
        const moduleLines = this._operations.map((op, index) => {
            return `            (${index}): ${op.to_torch()}`;
        });

        return `nn.Sequential(\n${moduleLines.join(',\n')}\n        )`;
    }

    to_torch_functional(input: string): string {
        if (this._operations.some(op => op.target !== "torch")) {
            throw new Error("All operations in sequence must be PyTorch operations");
        }

        // Generate multi-line code with each operation on its own line
        const lines = this._operations.map(op => {
            return `        ${op.to_torch_functional(input)}`;
        });

        return lines.join('\n');
    }

    private shapeMatch(op1: Op<T>, op2: Op<T>): boolean {
        if (!op1.outShape || !op2.inShape) {
            return false;
        }
        
        if (op1.outShape.length !== op2.inShape.length) {
            return false;
        }

        return op1.outShape.every((dim, index) => dim === op2.inShape[index]);
    }

    findById(id: string): Op<T> | undefined {
        return this._operations.find(op => op.id === id);
    }

    push(op: Op<T>): string {
        // Check shape compatibility before any modifications
        if (this._operations.length > 0) {
            const lastOp = this._operations[this._operations.length - 1];
            if (!this.shapeMatch(lastOp, op)) {
                throw new Error(`Shape mismatch: Cannot connect output shape [${lastOp.outShape}] to input shape [${op.inShape}]`);
            }
            // Set up links only after shape check passes
            lastOp.next = op;
            op.prev = lastOp;
        }

        // Add to array only after all validations pass
        this._operations.push(op);
        
        // Update Seq's shapes
        this._outShape = op.outShape;
        
        return op.id;
    }

    pop(): Op<T> | undefined {
        if (this._operations.length <= 1) {
            throw new Error("Cannot pop from sequence with only one operation");
        }

        const lastOp = this._operations[this._operations.length - 1];
        // Get the new last op's outShape before removing
        const newLastOp = this._operations[this._operations.length - 2];
        
        if (this.remove(lastOp.id)) {
            // Update outShape to the new last operation's outShape
            this._outShape = newLastOp.outShape;
            return lastOp;
        }
        return undefined;
    }

    insert(op: Op<T>, index: number): string {
        if (index < 0 || index > this._operations.length) {
            throw new Error("Index out of bounds");
        }

        if (index === this._operations.length) {
            return this.push(op);
        }

        // Check shape compatibility with previous op
        const prevOp = index > 0 ? this._operations[index - 1] : null;
        const nextOp = this._operations[index];

        if (prevOp && !this.shapeMatch(prevOp, op)) {
            throw new Error(`Shape mismatch: Cannot connect output shape [${prevOp.outShape}] to input shape [${op.inShape}]`);
        }

        // Check shape compatibility with next op
        if (!this.shapeMatch(op, nextOp)) {
            throw new Error(`Shape mismatch: Cannot connect output shape [${op.outShape}] to input shape [${nextOp.inShape}]`);
        }

        // Update links after shape checks pass
        if (prevOp) {
            prevOp.next = op;
            op.prev = prevOp;
        }

        op.next = nextOp;
        nextOp.prev = op;

        // Insert into array
        this._operations.splice(index, 0, op);

        // Update Seq's shapes if inserting at the beginning
        if (index === 0) {
            this._inShape = op.inShape;
        }
        
        return op.id;
    }

    remove(id: string): boolean {
        if (this._operations.length <= 1) {
            throw new Error("Cannot remove from sequence with only one operation");
        }

        const opIndex = this._operations.findIndex(op => op.id === id);
        if (opIndex === -1) {
            return false;  // Operation not found
        }

        const op = this._operations[opIndex];
        const prevOp = op.prev;
        const nextOp = op.next;
        
        // Check if the remaining operations can be connected
        if (prevOp && nextOp && !this.shapeMatch(prevOp, nextOp)) {
            throw new Error(`Shape mismatch: Cannot remove op as it would create invalid connection between output shape [${prevOp.outShape}] and input shape [${nextOp.inShape}]`);
        }

        // Update links after shape check passes
        if (prevOp) {
            prevOp.next = nextOp;
        }
        if (nextOp) {
            nextOp.prev = prevOp;
        }

        // Remove from array
        this._operations.splice(opIndex, 1);

        // Update Seq's shapes
        if (opIndex === 0) {
            // If removing first operation, update inShape
            this._inShape = this._operations[0].inShape;
        }
        if (opIndex === this._operations.length - 1) {
            // If removing last operation, update outShape
            this._outShape = this._operations[this._operations.length - 1].outShape;
        }

        return true;
    }

    [Symbol.iterator](): Iterator<Op<T>> {
        let index = 0;
        
        return {
            next: (): IteratorResult<Op<T>> => {
                if (index < this._operations.length) {
                    return {
                        value: this._operations[index++],
                        done: false
                    };
                } else {
                    return {
                        value: null as any,
                        done: true
                    };
                }
            }
        };
    }

    // Additional helper methods
    get length(): number {
        return this._operations.length;
    }

    get operations(): Op<T>[] {
        return [...this._operations];
    }
}
