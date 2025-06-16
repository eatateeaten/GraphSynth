import { Tensor } from '../DAGCompiler/tensor';
import { Copy } from '../DAGCompiler/branch_op';
import { PointwiseReduce } from '../DAGCompiler/reduce_op';
import { DotOp, CrossOp } from '../DAGCompiler/merge_op';

describe('Node Connection Methods - Part 2', () => {
    describe('Copy Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const copy = new Copy('00000000-0000-4000-8000-000000000001', 2, {copies: 2});
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [4, 10], 'input');

            // Add prev connection
            copy.addPrev(inputTensor, [4, 10]);

            // Verify connection
            expect(copy.prevs[0]).toBe(inputTensor);
            expect(copy.inShapes[0]).toEqual([4, 10]);

            // Verify output shapes updated
            expect(copy.outShapes).toHaveLength(2);
            expect(copy.outShapes[0]).toEqual([4, 10]);
            expect(copy.outShapes[1]).toEqual([4, 10]);

            // Delete prev connection
            copy.deletePrev();

            // Verify disconnection
            expect(copy.prevs[0]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const copy = new Copy('00000000-0000-4000-8000-000000000001', 2, {copies: 2});
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [4, 10], 'input');
            const output1 = new Tensor('00000000-0000-4000-8000-000000000003', [4, 10], 'y1');
            const output2 = new Tensor('00000000-0000-4000-8000-000000000004', [4, 10], 'y2');

            // Setup copy node with input
            copy.addPrev(inputTensor, [4, 10]);

            // Add next connections
            copy.addNext(output1, 0);
            copy.addNext(output2, 1);

            // Verify connections
            expect(copy.nexts[0]).toBe(output1);
            expect(copy.nexts[1]).toBe(output2);

            // Delete next connections
            copy.deleteNext(0);
            copy.deleteNext(1);

            // Verify disconnections
            expect(copy.nexts[0]).toBeNull();
            expect(copy.nexts[1]).toBeNull();
        });
    });

    describe('PointwiseReduce Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const reduce = new PointwiseReduce('00000000-0000-4000-8000-000000000001', 'Add', 2, {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 10], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [3, 10], 'x2');

            // Add prev connections
            reduce.addPrev(input1, [3, 10], 0);
            reduce.addPrev(input2, [3, 10], 1);

            // Verify connections
            expect(reduce.prevs[0]).toBe(input1);
            expect(reduce.prevs[1]).toBe(input2);
            expect(reduce.inShapes[0]).toEqual([3, 10]);
            expect(reduce.inShapes[1]).toEqual([3, 10]);

            // Verify output shape updated
            expect(reduce.outShapes[0]).toEqual([3, 10]);

            // Delete prev connections
            reduce.deletePrev(0);
            reduce.deletePrev(1);

            // Verify disconnections
            expect(reduce.prevs[0]).toBeNull();
            expect(reduce.prevs[1]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const reduce = new PointwiseReduce('00000000-0000-4000-8000-000000000001', 'Add', 2, {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 10], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [3, 10], 'x2');
            const output = new Tensor('00000000-0000-4000-8000-000000000004', [3, 10], 'y');

            // Setup reduce node with inputs
            reduce.addPrev(input1, [3, 10], 0);
            reduce.addPrev(input2, [3, 10], 1);

            // Add next connection
            reduce.addNext(output);

            // Verify connection
            expect(reduce.nexts[0]).toBe(output);

            // Delete next connection
            reduce.deleteNext();

            // Verify disconnection
            expect(reduce.nexts[0]).toBeNull();
        });

        test('should reject mismatched input shapes', () => {
            // Create nodes
            const reduce = new PointwiseReduce('00000000-0000-4000-8000-000000000001', 'Add', 2, {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 10], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [4, 10], 'x2');

            // Add first input
            reduce.addPrev(input1, [3, 10], 0);

            // Adding mismatched second input should throw
            expect(() => {
                reduce.addPrev(input2, [4, 10], 1);
            }).toThrow();
        });
    });

    describe('DotOp Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const dot = new DotOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 4], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [4, 5], 'x2');

            // Add prev connections
            dot.addPrev(input1, [3, 4], 0);
            dot.addPrev(input2, [4, 5], 1);

            // Verify connections
            expect(dot.prevs[0]).toBe(input1);
            expect(dot.prevs[1]).toBe(input2);
            expect(dot.inShapes[0]).toEqual([3, 4]);
            expect(dot.inShapes[1]).toEqual([4, 5]);

            // Verify output shape updated
            expect(dot.outShapes[0]).toEqual([3, 5]);

            // Delete prev connections
            dot.deletePrev(0);
            dot.deletePrev(1);

            // Verify disconnections
            expect(dot.prevs[0]).toBeNull();
            expect(dot.prevs[1]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const dot = new DotOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 4], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [4, 5], 'x2');
            const output = new Tensor('00000000-0000-4000-8000-000000000004', [3, 5], 'y');

            // Setup dot node with inputs
            dot.addPrev(input1, [3, 4], 0);
            dot.addPrev(input2, [4, 5], 1);

            // Add next connection
            dot.addNext(output);

            // Verify connection
            expect(dot.nexts[0]).toBe(output);

            // Delete next connection
            dot.deleteNext();

            // Verify disconnection
            expect(dot.nexts[0]).toBeNull();
        });

        test('should reject incompatible dimensions', () => {
            // Create nodes
            const dot = new DotOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 4], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [5, 6], 'x2');

            // Add first input
            dot.addPrev(input1, [3, 4], 0);

            // Adding incompatible second input should throw
            expect(() => {
                dot.addPrev(input2, [5, 6], 1);
            }).toThrow();
        });
    });

    describe('CrossOp Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const cross = new CrossOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [2, 3], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [2, 3], 'x2');

            // Add prev connections
            cross.addPrev(input1, [2, 3], 0);
            cross.addPrev(input2, [2, 3], 1);

            // Verify connections
            expect(cross.prevs[0]).toBe(input1);
            expect(cross.prevs[1]).toBe(input2);
            expect(cross.inShapes[0]).toEqual([2, 3]);
            expect(cross.inShapes[1]).toEqual([2, 3]);

            // Verify output shape updated
            expect(cross.outShapes[0]).toEqual([2, 3]);

            // Delete prev connections
            cross.deletePrev(0);
            cross.deletePrev(1);

            // Verify disconnections
            expect(cross.prevs[0]).toBeNull();
            expect(cross.prevs[1]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const cross = new CrossOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [2, 3], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [2, 3], 'x2');
            const output = new Tensor('00000000-0000-4000-8000-000000000004', [2, 3], 'y');

            // Setup cross node with inputs
            cross.addPrev(input1, [2, 3], 0);
            cross.addPrev(input2, [2, 3], 1);

            // Add next connection
            cross.addNext(output);

            // Verify connection
            expect(cross.nexts[0]).toBe(output);

            // Delete next connection
            cross.deleteNext();

            // Verify disconnection
            expect(cross.nexts[0]).toBeNull();
        });

        test('should enforce 3D vector requirement', () => {
            // Create nodes
            const cross = new CrossOp('00000000-0000-4000-8000-000000000001', {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [2, 3], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [2, 4], 'x2');

            // Add first input
            cross.addPrev(input1, [2, 3], 0);

            // Adding non-3D second input should throw
            expect(() => {
                cross.addPrev(input2, [2, 4], 1);
            }).toThrow();
        });
    });
});
