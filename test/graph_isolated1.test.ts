import { Tensor } from '../DAGCompiler/tensor';
import { Split } from '../DAGCompiler/branch_op';
import { Concat } from '../DAGCompiler/reduce_op';

describe('Node Connection Methods', () => {
    describe('Tensor Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const tensor = new Tensor('00000000-0000-4000-8000-000000000001', [3, 32, 32], 'x');
            const prevTensor = new Tensor('00000000-0000-4000-8000-000000000002', [3, 32, 32], 'input');

            // Add prev connection
            tensor.addPrev(prevTensor);

            // Verify connection
            expect(tensor.prevs[0]).toBe(prevTensor);
            expect(tensor.inShapes[0]).toEqual([3, 32, 32]);

            // Delete prev connection
            tensor.deletePrev();

            // Verify disconnection
            expect(tensor.prevs[0]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const tensor = new Tensor('00000000-0000-4000-8000-000000000001', [3, 32, 32], 'x');
            const nextTensor = new Tensor('00000000-0000-4000-8000-000000000002', [3, 32, 32], 'output');

            // Add next connection
            tensor.addNext(nextTensor);

            // Verify connection
            expect(tensor.nexts[0]).toBe(nextTensor);

            // Delete next connection
            tensor.deleteNext();

            // Verify disconnection
            expect(tensor.nexts[0]).toBeNull();
        });
    });

    describe('Split Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const split = new Split('00000000-0000-4000-8000-000000000001', 0, [2, 2], {});
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [4, 10], 'input');

            // Add prev connection
            split.addPrev(inputTensor, [4, 10]);

            // Verify connection
            expect(split.prevs[0]).toBe(inputTensor);
            expect(split.inShapes[0]).toEqual([4, 10]);

            // Verify output shapes updated
            expect(split.outShapes).toHaveLength(2);
            expect(split.outShapes[0]).toEqual([2, 10]);
            expect(split.outShapes[1]).toEqual([2, 10]);

            // Delete prev connection
            split.deletePrev();

            // Verify disconnection
            expect(split.prevs[0]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const split = new Split('00000000-0000-4000-8000-000000000001', 0, [2, 2], {});
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [4, 10], 'input');
            const output1 = new Tensor('00000000-0000-4000-8000-000000000003', [2, 10], 'y1');
            const output2 = new Tensor('00000000-0000-4000-8000-000000000004', [2, 10], 'y2');

            // Setup split node with input
            split.addPrev(inputTensor, [4, 10]);

            // Add next connections
            split.addNext(output1, 0);
            split.addNext(output2, 1);

            // Verify connections
            expect(split.nexts[0]).toBe(output1);
            expect(split.nexts[1]).toBe(output2);

            // Delete next connections
            split.deleteNext(0);
            split.deleteNext(1);

            // Verify disconnections
            expect(split.nexts[0]).toBeNull();
            expect(split.nexts[1]).toBeNull();
        });
    });

    describe('Concat Node', () => {
        test('should add and delete prev connections', () => {
            // Create nodes
            const concat = new Concat('00000000-0000-4000-8000-000000000001', 0, 2, {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 10], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [5, 10], 'x2');

            // Add prev connections
            concat.addPrev(input1, [3, 10], 0);
            concat.addPrev(input2, [5, 10], 1);

            // Verify connections
            expect(concat.prevs[0]).toBe(input1);
            expect(concat.prevs[1]).toBe(input2);
            expect(concat.inShapes[0]).toEqual([3, 10]);
            expect(concat.inShapes[1]).toEqual([5, 10]);

            // Verify output shape updated
            expect(concat.outShapes[0]).toEqual([8, 10]);

            // Delete prev connections
            concat.deletePrev(0);
            concat.deletePrev(1);

            // Verify disconnections
            expect(concat.prevs[0]).toBeNull();
            expect(concat.prevs[1]).toBeNull();
        });

        test('should add and delete next connections', () => {
            // Create nodes
            const concat = new Concat('00000000-0000-4000-8000-000000000001', 0, 2, {});
            const input1 = new Tensor('00000000-0000-4000-8000-000000000002', [3, 10], 'x1');
            const input2 = new Tensor('00000000-0000-4000-8000-000000000003', [5, 10], 'x2');
            const output = new Tensor('00000000-0000-4000-8000-000000000004', [8, 10], 'y');

            // Setup concat node with inputs
            concat.addPrev(input1, [3, 10], 0);
            concat.addPrev(input2, [5, 10], 1);

            // Add next connection
            concat.addNext(output);

            // Verify connection
            expect(concat.nexts[0]).toBe(output);

            // Delete next connection
            concat.deleteNext();

            // Verify disconnection
            expect(concat.nexts[0]).toBeNull();
        });
    });
});
