import { Tensor, Op, GraphNode, Concat, Split } from '../front/graph';
import { describe, test, expect, beforeEach } from '@jest/globals';

// Helper function to generate IDs
function generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0,
            v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

describe('GraphNode Connections', () => {
    let tensorA: Tensor;
    let tensorB: Tensor;
    let basicOp: Op;
    let mergeOp: Concat;
    let splitOp: Split;

    // Use consistent shapes
    const shape1 = [1, 64, 112, 112];
    const shape2 = [1, 128, 56, 56];
    const shape3 = [1, 256, 28, 28];

    beforeEach(() => {
        // Create tensors with matching shapes
        tensorA = new Tensor(generateId(), shape1, "torch");
        tensorB = new Tensor(generateId(), shape1, "torch");
        
        // Create basic op with matching shapes
        basicOp = new Op(
            generateId(),
            shape1, // inShape matches tensorA's outShape
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
        
        // Create mergeOp with matching shapes
        mergeOp = new Concat(
            generateId(),
            [shape1, shape1], // inShapes - both inputs match tensorA/B
            "torch",
            { dim: 1 } // concat along channel dimension
        );
        
        // Create splitOp with matching shapes
        splitOp = new Split(
            generateId(),
            shape2, // inShape
            "torch",
            { sections: [128, 128], dim: 1 } // Split along channel dimension
        );
    });

    describe('Tensor connections', () => {
        it('Tensor to Tensor connection', () => {
            // Test bidirectional connection
            tensorB.connectSource(tensorA);
            expect(tensorB.prev).toBe(tensorA);
            expect(tensorA.next).toBe(tensorB);
            
            // Test disconnection
            tensorB.disconnectSource();
            expect(tensorB.prev).toBeNull();
            expect(tensorA.next).toBeNull();
        });

        it('Tensor to Op connection', () => {
            // Test connection
            basicOp.connectSource(tensorA);
            expect(basicOp.prev).toBe(tensorA);
            expect(tensorA.next).toBe(basicOp);
            
            // Test disconnection
            basicOp.disconnectSource();
            expect(basicOp.prev).toBeNull();
            expect(tensorA.next).toBeNull();
        });

        it('Tensor to MergeOp connection', () => {
            // Test connection - connect to first input
            tensorA.connectSink(mergeOp, undefined, 0);
            expect(mergeOp.prev).toBe(tensorA);
            expect(tensorA.next).toBe(mergeOp);
            
            // Test disconnection
            mergeOp.disconnectSource();
            expect(mergeOp.prev).toBeNull();
            expect(tensorA.next).toBeNull();
        });

        it('Tensor to BranchOp connection', () => {
            const tempTensor = new Tensor(generateId(), shape2, "torch");
            
            // Test connection
            splitOp.connectSource(tempTensor);
            expect(splitOp.prev).toBe(tempTensor);
            expect(tempTensor.next).toBe(splitOp);
            
            // Test disconnection
            splitOp.disconnectSource();
            expect(splitOp.prev).toBeNull();
            expect(tempTensor.next).toBeNull();
        });
    });

    describe('Op connections', () => {
        it('Op to Tensor connection', () => {
            const outTensor = new Tensor(generateId(), shape2, "torch");
            
            // Test connection
            outTensor.connectSource(basicOp);
            expect(outTensor.prev).toBe(basicOp);
            expect(basicOp.next).toBe(outTensor);
            
            // Test disconnection
            outTensor.disconnectSource();
            expect(outTensor.prev).toBeNull();
            expect(basicOp.next).toBeNull();
        });

        it('Op to BranchOp connection', () => {
            // Test connection
            splitOp.connectSource(basicOp);
            expect(splitOp.prev).toBe(basicOp);
            expect(basicOp.next).toBe(splitOp);
            
            // Test disconnection
            splitOp.disconnectSource();
            expect(splitOp.prev).toBeNull();
            expect(basicOp.next).toBeNull();
        });
    });

    describe('MergeOp connections', () => {
        it('Multiple sources to MergeOp connection', () => {
            // Create another tensor with the same shape for the second input
            const tensorC = new Tensor(generateId(), shape1, "torch");
            
            // Connect first source
            tensorA.connectSink(mergeOp, undefined, 0);
            expect(mergeOp.prev).toBe(tensorA);
            expect(tensorA.next).toBe(mergeOp);
            
            // Connect second source
            tensorC.connectSink(mergeOp, undefined, 1);
            expect(tensorC.next).toBe(mergeOp);
            
            // Test disconnection
            mergeOp.disconnectSource();
            expect(mergeOp.prev).toBeNull();
            expect(tensorA.next).toBeNull();
            expect(tensorC.next).toBeNull();
        });

        it('MergeOp to Op connection', () => {
            // Create special op with input shape matching mergeOp's outShape
            const nextOp = new Op(
                generateId(),
                [1, 128, 112, 112], // inShape matching mergeOp's outShape
                "torch",
                "Conv2D",
                {
                    in_channels: 128,
                    out_channels: 256,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1
                }
            );
            
            // Connect sources to MergeOp first
            const tensorC = new Tensor(generateId(), shape1, "torch");
            mergeOp.connectSource(tensorA, 0);
            mergeOp.connectSource(tensorC, 1);
            
            // Connect MergeOp to next Op
            nextOp.connectSource(mergeOp);
            expect(nextOp.prev).toBe(mergeOp);
            expect(mergeOp.next).toBe(nextOp);
            
            // Test disconnection
            nextOp.disconnectSource();
            expect(nextOp.prev).toBeNull();
            expect(mergeOp.next).toBeNull();
        });

        it('should disconnect specific inputs from MergeOp with index', () => {
            const shapes = [[3, 3], [3, 3], [3, 3]];
            const t1 = new Tensor(generateId(), shapes[0], 'tensor1');
            const t2 = new Tensor(generateId(), shapes[1], 'tensor2');
            const t3 = new Tensor(generateId(), shapes[2], 'tensor3');
            const concat = new Concat(generateId(), shapes, 'torch', { dim: 1 });
            const output = new Tensor(generateId(), [3, 9], 'output');

            // Connect all inputs
            t1.connectSink(concat, undefined, 0);
            t2.connectSink(concat, undefined, 1);
            t3.connectSink(concat, undefined, 2);
            concat.connectSink(output);

            // Disconnect the second input specifically
            concat.disconnectSource(1);

            // Check connections
            expect(t1.next).toBe(concat);
            expect(t2.next).toBeNull();
            expect(t3.next).toBe(concat);
            expect(concat._prevs[0]).toBe(t1);
            expect(concat._prevs[1]).toBeNull();
            expect(concat._prevs[2]).toBe(t3);
            expect(concat.next).toBe(output);
            expect(output.prev).toBe(concat);
        });
    });

    describe('BranchOp connections', () => {
        it('BranchOp to multiple sinks connection', () => {
            // Create tensors with shape matching the output shape of splitOp
            const outTensor1 = new Tensor(generateId(), [1, 128, 56, 56], "torch");
            const outTensor2 = new Tensor(generateId(), [1, 128, 56, 56], "torch");
            
            // Connect a tensor as input to the BranchOp first
            const inputTensor = new Tensor(generateId(), shape2, "torch");
            splitOp.connectSource(inputTensor);
            
            // Connect BranchOp to first output
            splitOp.connectSink(outTensor1, 0);
            expect(splitOp.nexts[0]).toBe(outTensor1);
            expect(outTensor1.prev).toBe(splitOp);
            
            // Connect BranchOp to second output
            splitOp.connectSink(outTensor2, 1);
            expect(splitOp.nexts[1]).toBe(outTensor2);
            expect(outTensor2.prev).toBe(splitOp);
            
            // Test disconnection
            splitOp.disconnectSink();
            expect(splitOp.next).toBeNull();
            expect(outTensor1.prev).toBeNull();
            expect(outTensor2.prev).toBeNull();
        });

        it('BranchOp to multiple Ops connection', () => {
            // Create ops with input shape matching the output shape of splitOp
            const nextOp1 = new Op(
                generateId(),
                [1, 128, 56, 56], // inShape matching splitOp's outShape
                "torch",
                "Conv2D", 
                {
                    in_channels: 128,
                    out_channels: 512,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1
                }
            );
            
            const nextOp2 = new Op(
                generateId(),
                [1, 128, 56, 56], // inShape matching splitOp's outShape
                "torch",
                "Conv2D",
                {
                    in_channels: 128,
                    out_channels: 512,
                    kernel_size: 3,
                    stride: 2,
                    padding: 1
                }
            );
            
            // Connect a tensor as input to the BranchOp first
            const inputTensor = new Tensor(generateId(), shape2, "torch");
            splitOp.connectSource(inputTensor);
            
            // Connect BranchOp to first output Op
            splitOp.connectSink(nextOp1, 0);
            expect(splitOp.nexts[0]).toBe(nextOp1);
            expect(nextOp1.prev).toBe(splitOp);
            
            // Connect BranchOp to second output Op
            splitOp.connectSink(nextOp2, 1);
            expect(splitOp.nexts[1]).toBe(nextOp2);
            expect(nextOp2.prev).toBe(splitOp);
            
            // Test disconnection
            splitOp.disconnectSink();
            expect(splitOp.next).toBeNull();
            expect(nextOp1.prev).toBeNull();
            expect(nextOp2.prev).toBeNull();
        });

        it('should disconnect specific outputs from BranchOp with index', () => {
            const inputShape = [6, 6];
            const outputShapes = [[6, 2], [6, 2], [6, 2]];
            const input = new Tensor(generateId(), inputShape, 'input');
            const split = new Split(generateId(), inputShape, 'torch', { sections: [2, 2, 2], dim: 1 });
            const t1 = new Tensor(generateId(), outputShapes[0], 'output1');
            const t2 = new Tensor(generateId(), outputShapes[1], 'output2');
            const t3 = new Tensor(generateId(), outputShapes[2], 'output3');

            // Connect all outputs
            input.connectSink(split);
            split.connectSink(t1, 0);
            split.connectSink(t2, 1);
            split.connectSink(t3, 2);

            // Disconnect the second output specifically
            split.disconnectSink(1);

            // Check connections
            expect(input.next).toBe(split);
            expect(split.prev).toBe(input);
            expect(split._nexts[0]).toBe(t1);
            expect(split._nexts[1]).toBeNull();
            expect(split._nexts[2]).toBe(t3);
            expect(t1.prev).toBe(split);
            expect(t2.prev).toBeNull();
            expect(t3.prev).toBe(split);
        });
    });

    describe('Complex connections', () => {
        it('Complex network with multiple connections', () => {
            // Create a more complex network:
            // tensor1 -> op1 -> tensor2 ─┐
            //                            └─> mergeOp2 -> tensor4 -> splitOp2 -> tensor5
            // tensor3 -----------------─┘                                    └─> tensor6

            const tensor1 = new Tensor(generateId(), shape1, "torch");
            const op1 = new Op(
                generateId(),
                shape1,
                "torch",
                "Conv2D",
                { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1 }
            );
            const tensor2 = new Tensor(generateId(), shape1, "torch");
            const tensor3 = new Tensor(generateId(), shape1, "torch");
            
            const mergeOp2 = new Concat(
                generateId(),
                [shape1, shape1],
                "torch",
                { dim: 1 }
            );
            
            const tensor4 = new Tensor(generateId(), [1, 128, 112, 112], "torch");
            
            const splitOp2 = new Split(
                generateId(),
                [1, 128, 112, 112],
                "torch",
                { sections: [64, 64], dim: 1 }
            );
            
            const tensor5 = new Tensor(generateId(), [1, 64, 112, 112], "torch");
            const tensor6 = new Tensor(generateId(), [1, 64, 112, 112], "torch");
            
            // Connect first path
            op1.connectSource(tensor1);
            tensor2.connectSource(op1);
            tensor2.connectSink(mergeOp2, undefined, 0);
            
            // Connect second path
            tensor3.connectSink(mergeOp2, undefined, 1);
            
            // Connect merged paths
            tensor4.connectSource(mergeOp2);
            splitOp2.connectSource(tensor4);
            
            // Connect split paths
            tensor5.connectSource(splitOp2, undefined, 0);
            tensor6.connectSource(splitOp2, undefined, 1);
            
            // Verify connections
            expect(op1.prev).toBe(tensor1);
            expect(tensor2.prev).toBe(op1);
            expect(mergeOp2.prev).toBe(tensor2);
            expect(tensor3.next).toBe(mergeOp2);
            expect(tensor4.prev).toBe(mergeOp2);
            expect(splitOp2.prev).toBe(tensor4);
            expect(tensor5.prev).toBe(splitOp2);
            expect(tensor6.prev).toBe(splitOp2);
            
            // Test shape validation
            const wrongShapeTensor = new Tensor(generateId(), [2, 32, 16, 16], "torch");
            expect(() => {
                op1.connectSource(wrongShapeTensor);
            }).toThrow(/Shape mismatch/);
        });
    });

    describe('Shape validation', () => {
        it('Should throw error when trying to connect incompatible shapes', () => {
            const incompatibleTensor = new Tensor(generateId(), [1, 32, 64, 64], "torch");
            
            expect(() => {
                basicOp.connectSource(incompatibleTensor);
            }).toThrow(/Shape mismatch/);
            
            expect(() => {
                incompatibleTensor.connectSink(mergeOp, undefined, 0);
            }).toThrow(/Shape mismatch/);
            
            const wrongInputForSplit = new Tensor(generateId(), [1, 32, 28, 28], "torch");
            expect(() => {
                splitOp.connectSource(wrongInputForSplit);
            }).toThrow(/Shape mismatch/);
        });

        it('Should throw error when connecting without required indices', () => {
            const tensor = new Tensor(generateId(), shape1, "torch");
            
            // MergeOp requires index
            expect(() => {
                tensor.connectSink(mergeOp);
            }).toThrow(/When connecting to a MergeOp, an input index must be specified/);
            
            // BranchOp requires index
            const branchTensor = new Tensor(generateId(), [1, 128, 56, 56], "torch");
            expect(() => {
                branchTensor.connectSource(splitOp);
            }).toThrow(/When connecting from a BranchOp, an output index must be specified/);
        });
    });
}); 