import { Op } from '../front/graph';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('Op', () => {
    let op: Op<number>;

    beforeEach(() => {
        op = new Op<number>(
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
    });

    test('constructor initializes properties correctly', () => {
        expect(op.inShape).toEqual([1, 3, 224, 224]);
        expect(op.outShape).toEqual([1, 64, 112, 112]);
        expect(op.target).toBe("torch");
        expect(op.opType).toBe("Conv2D");
        expect(op.params).toEqual({
            in_channels: 3,
            out_channels: 64,
            kernel_size: 7,
            stride: 2,
            padding: 3
        });
        expect(op.id).toBeDefined();
        expect(typeof op.id).toBe('string');
    });

    test('to_torch generates correct PyTorch code', () => {
        const torchCode = op.to_torch();
        expect(torchCode).toBe('nn.Conv2d(3, 64, 7, stride=2, padding=3, dilation=1, groups=1, bias=true, padding_mode=\'zeros\')');
    });

    test('to_torch throws error for non-PyTorch operations', () => {
        const nonTorchOp = new Op<number>(
            [1, 3, 224, 224],
            [1, 64, 112, 112],
            "tensorflow",
            "Conv2D",
            { filters: 64, kernel_size: 7 }
        );

        expect(() => nonTorchOp.to_torch()).toThrow("Operation is not a PyTorch operation");
    });

    test('params getter returns a copy of parameters', () => {
        const params = op.params;
        params.in_channels = 10; // Modifying the copy
        expect(op.params.in_channels).toBe(3); // Original should be unchanged
    });

    test('prev and next setters work correctly', () => {
        const prevOp = new Op<number>(
            [1, 1, 448, 448],
            [1, 3, 224, 224],
            "torch",
            "Conv2D",
            { in_channels: 1, out_channels: 3, kernel_size: 3 }
        );

        const nextOp = new Op<number>(
            [1, 64, 112, 112],
            [1, 64, 112, 112],
            "torch",
            "ReLU",
            {}
        );

        op.prev = prevOp;
        op.next = nextOp;

        expect(op.prev).toBe(prevOp);
        expect(op.next).toBe(nextOp);
    });
}); 