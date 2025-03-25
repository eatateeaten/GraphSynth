import { Op, Seq } from '../front/graph';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('Seq', () => {
    let conv1: Op<number>;
    let relu1: Op<number>;
    let conv2: Op<number>;
    let seq: Seq<number>;

    beforeEach(() => {
        // Create a Conv2D operation
        conv1 = new Op<number>(
            [1, 3, 224, 224],  // inShape
            [1, 64, 112, 112], // outShape
            "torch",
            "Conv2D",
            {
                in_channels: 3,
                out_channels: 64,
                kernel_size: 7,
                stride: 2,
                padding: 3
            }
        );

        // Create a ReLU operation
        relu1 = new Op<number>(
            [1, 64, 112, 112], // inShape
            [1, 64, 112, 112], // outShape
            "torch",
            "ReLU",
            {}
        );

        // Create another Conv2D operation
        conv2 = new Op<number>(
            [1, 64, 112, 112], // inShape
            [1, 128, 56, 56],  // outShape
            "torch",
            "Conv2D",
            {
                in_channels: 64,
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1
            }
        );

        // Initialize sequence with conv1
        seq = new Seq<number>(conv1);
    });

    test('constructor initializes with single operation', () => {
        expect(seq.length).toBe(1);
        expect(seq.operations).toEqual([conv1]);
        expect(seq.inShape).toEqual([1, 3, 224, 224]);
        expect(seq.outShape).toEqual([1, 64, 112, 112]);
        expect(seq.target).toBe("torch");
        expect(seq.opType).toBe("Seq");
    });

    test('push adds operation to sequence', () => {
        seq.push(relu1);
        expect(seq.length).toBe(2);
        expect(seq.operations).toEqual([conv1, relu1]);
        expect(seq.outShape).toEqual([1, 64, 112, 112]);
        
        seq.push(conv2);
        expect(seq.length).toBe(3);
        expect(seq.operations).toEqual([conv1, relu1, conv2]);
        expect(seq.outShape).toEqual([1, 128, 56, 56]);
    });

    test('push throws error on shape mismatch', () => {
        const invalidOp = new Op<number>(
            [1, 32, 112, 112], // Wrong input shape
            [1, 64, 56, 56],
            "torch",
            "Conv2D",
            { in_channels: 32, out_channels: 64, kernel_size: 3 }
        );

        expect(() => seq.push(invalidOp)).toThrow("Shape mismatch");
    });

    test('pop removes and returns last operation', () => {
        seq.push(relu1);
        seq.push(conv2);
        
        const popped = seq.pop();
        expect(popped).toBe(conv2);
        expect(seq.length).toBe(2);
        expect(seq.operations).toEqual([conv1, relu1]);
        expect(seq.outShape).toEqual([1, 64, 112, 112]);
    });

    test('pop throws error on single operation sequence', () => {
        expect(() => seq.pop()).toThrow("Cannot pop from sequence with only one operation");
    });

    test('insert adds operation at specified index', () => {
        seq.push(conv2);
        seq.insert(relu1, 1);
        
        expect(seq.length).toBe(3);
        expect(seq.operations).toEqual([conv1, relu1, conv2]);
        expect(seq.outShape).toEqual([1, 128, 56, 56]);
    });

    test('insert throws error on shape mismatch', () => {
        seq.push(conv2);
        const invalidOp = new Op<number>(
            [1, 32, 112, 112],
            [1, 64, 56, 56],
            "torch",
            "Conv2D",
            { in_channels: 32, out_channels: 64, kernel_size: 3 }
        );

        expect(() => seq.insert(invalidOp, 1)).toThrow("Shape mismatch");
    });

    test('remove deletes operation by id', () => {
        seq.push(relu1);
        seq.push(conv2);
        
        const removed = seq.remove(relu1.id);
        expect(removed).toBe(true);
        expect(seq.length).toBe(2);
        expect(seq.operations).toEqual([conv1, conv2]);
    });

    test('remove throws error on single operation sequence', () => {
        expect(() => seq.remove(conv1.id)).toThrow("Cannot remove from sequence with only one operation");
    });

    test('remove throws error on shape mismatch', () => {
        // Create a sequence where removing the middle operation would create a shape mismatch
        const op1 = new Op<number>(
            [1, 3, 224, 224],
            [1, 64, 112, 112],
            "torch",
            "Conv2D",
            { in_channels: 3, out_channels: 64, kernel_size: 7 }
        );

        const op2 = new Op<number>(
            [1, 64, 112, 112],
            [1, 32, 56, 56],
            "torch",
            "Conv2D",
            { in_channels: 64, out_channels: 32, kernel_size: 3 }
        );

        const op3 = new Op<number>(
            [1, 32, 56, 56],
            [1, 16, 28, 28],
            "torch",
            "Conv2D",
            { in_channels: 32, out_channels: 16, kernel_size: 3 }
        );

        const testSeq = new Seq<number>(op1);
        testSeq.push(op2);
        testSeq.push(op3);

        expect(() => testSeq.remove(op2.id)).toThrow("Shape mismatch");
    });

    test('findById returns correct operation', () => {
        seq.push(relu1);
        seq.push(conv2);

        expect(seq.findById(conv1.id)).toBe(conv1);
        expect(seq.findById(relu1.id)).toBe(relu1);
        expect(seq.findById(conv2.id)).toBe(conv2);
        expect(seq.findById('nonexistent-id')).toBeUndefined();
    });

    test('iterator works correctly', () => {
        seq.push(relu1);
        seq.push(conv2);

        const operations = [];
        for (const op of seq) {
            operations.push(op);
        }

        expect(operations).toEqual([conv1, relu1, conv2]);
    });
}); 