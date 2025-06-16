import { Tensor } from '../DAGCompiler/tensor';
import { Op } from '../DAGCompiler/op';
import { g_GraphConfig } from '../DAGCompiler/config';

// Set the target to Torch for shape inference
beforeAll(() => {
    g_GraphConfig.target = 'Torch';
});

describe('Op Node Connection Methods with Different Operation Types', () => {
    describe('Linear Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const linear = new Op('00000000-0000-4000-8000-000000000001', 'Linear', {
                input_features: 100,
                output_features: 50,
                bias: true
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [32, 100], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [32, 50], 'output');

            // Add prev connection
            linear.addPrev(inputTensor, [32, 100]);

            // Verify prev connection
            expect(linear.prevs[0]).toBe(inputTensor);
            expect(linear.inShapes[0]).toEqual([32, 100]);

            // Verify output shape updated
            expect(linear.outShapes[0]).toEqual([32, 50]);

            // Add next connection
            linear.addNext(outputTensor);

            // Verify next connection
            expect(linear.nexts[0]).toBe(outputTensor);

            // Delete connections
            linear.deleteNext();
            linear.deletePrev();

            // Verify disconnections
            expect(linear.prevs[0]).toBeNull();
            expect(linear.nexts[0]).toBeNull();
        });

        test('should reject incompatible input shapes', () => {
            // Create nodes
            const linear = new Op('00000000-0000-4000-8000-000000000001', 'Linear', {
                input_features: 100,
                output_features: 50,
                bias: true
            });
            const incompatibleTensor = new Tensor('00000000-0000-4000-8000-000000000002', [32, 64], 'input');

            // Attempt to add prev connection with wrong input feature size
            expect(() => {
                linear.addPrev(incompatibleTensor, [32, 64]);
            }).toThrow();
        });
    });

    describe('Conv2D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const conv = new Op('00000000-0000-4000-8000-000000000001', 'Conv2D', {
                in_channels: 3,
                out_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 3, 32, 32], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 32, 32], 'output');

            // Add prev connection
            conv.addPrev(inputTensor, [1, 3, 32, 32]);

            // Verify prev connection
            expect(conv.prevs[0]).toBe(inputTensor);
            expect(conv.inShapes[0]).toEqual([1, 3, 32, 32]);

            // Verify output shape updated
            expect(conv.outShapes[0]).toEqual([1, 16, 32, 32]);

            // Add next connection
            conv.addNext(outputTensor);

            // Verify next connection
            expect(conv.nexts[0]).toBe(outputTensor);

            // Delete connections
            conv.deleteNext();
            conv.deletePrev();

            // Verify disconnections
            expect(conv.prevs[0]).toBeNull();
            expect(conv.nexts[0]).toBeNull();
        });

        test('should reject incompatible input shapes', () => {
            // Create nodes
            const conv = new Op('00000000-0000-4000-8000-000000000001', 'Conv2D', {
                in_channels: 3,
                out_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1
            });
            
            // Tensor with wrong number of channels
            const incompatibleTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 4, 32, 32], 'input');

            // Attempt to add prev connection with wrong channel count
            expect(() => {
                conv.addPrev(incompatibleTensor, [1, 4, 32, 32]);
            }).toThrow();
            
            // Tensor with wrong dimensionality
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 3, 32], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                conv.addPrev(wrongDimTensor, [1, 3, 32]);
            }).toThrow();
        });
    });

    describe('BatchNorm2D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const batchnorm = new Op('00000000-0000-4000-8000-000000000001', 'BatchNorm2D', {
                num_features: 16
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16, 32, 32], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 32, 32], 'output');

            // Add prev connection
            batchnorm.addPrev(inputTensor, [1, 16, 32, 32]);

            // Verify prev connection
            expect(batchnorm.prevs[0]).toBe(inputTensor);
            expect(batchnorm.inShapes[0]).toEqual([1, 16, 32, 32]);

            // Verify output shape unchanged (shape preservation)
            expect(batchnorm.outShapes[0]).toEqual([1, 16, 32, 32]);

            // Add next connection
            batchnorm.addNext(outputTensor);

            // Verify next connection
            expect(batchnorm.nexts[0]).toBe(outputTensor);

            // Delete connections
            batchnorm.deleteNext();
            batchnorm.deletePrev();

            // Verify disconnections
            expect(batchnorm.prevs[0]).toBeNull();
            expect(batchnorm.nexts[0]).toBeNull();
        });

        test('should reject incompatible input shapes', () => {
            // Create nodes
            const batchnorm = new Op('00000000-0000-4000-8000-000000000001', 'BatchNorm2D', {
                num_features: 16
            });
            
            // Tensor with wrong number of features
            const incompatibleTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 8, 32, 32], 'input');

            // Attempt to add prev connection with wrong feature count
            expect(() => {
                batchnorm.addPrev(incompatibleTensor, [1, 8, 32, 32]);
            }).toThrow();
            
            // Tensor with wrong dimensionality
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                batchnorm.addPrev(wrongDimTensor, [1, 16]);
            }).toThrow();
        });
    });

    describe('MaxPool2D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const pool = new Op('00000000-0000-4000-8000-000000000001', 'MaxPool2D', {
                kernel_size: 2,
                stride: 2
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16, 32, 32], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 16, 16], 'output');

            // Add prev connection
            pool.addPrev(inputTensor, [1, 16, 32, 32]);

            // Verify prev connection
            expect(pool.prevs[0]).toBe(inputTensor);
            expect(pool.inShapes[0]).toEqual([1, 16, 32, 32]);

            // Verify output shape updated (halved dimensions)
            expect(pool.outShapes[0]).toEqual([1, 16, 16, 16]);

            // Add next connection
            pool.addNext(outputTensor);

            // Verify next connection
            expect(pool.nexts[0]).toBe(outputTensor);

            // Delete connections
            pool.deleteNext();
            pool.deletePrev();

            // Verify disconnections
            expect(pool.prevs[0]).toBeNull();
            expect(pool.nexts[0]).toBeNull();
        });

        test('should reject incompatible input shapes', () => {
            // Create nodes
            const pool = new Op('00000000-0000-4000-8000-000000000001', 'MaxPool2D', {
                kernel_size: 2,
                stride: 2
            });
            
            // Tensor with wrong dimensionality (expecting 4D tensor for MaxPool2D)
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000002', [16, 32], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                pool.addPrev(wrongDimTensor, [16, 32]);
            }).toThrow();
            
            // Tensor with dimension too small for pooling
            const smallDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 1, 1], 'input');
            
            // Attempt to add prev connection with dimensions too small for the kernel size
            expect(() => {
                pool.addPrev(smallDimTensor, [1, 16, 1, 1]);
            }).toThrow();
        });
    });

    describe('Flatten Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const flatten = new Op('00000000-0000-4000-8000-000000000001', 'Flatten', {
                start_dim: 1,
                end_dim: -1
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16, 16, 16], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 4096], 'output');

            // Add prev connection
            flatten.addPrev(inputTensor, [1, 16, 16, 16]);

            // Verify prev connection
            expect(flatten.prevs[0]).toBe(inputTensor);
            expect(flatten.inShapes[0]).toEqual([1, 16, 16, 16]);

            // Verify output shape updated (flattened dimensions)
            expect(flatten.outShapes[0]).toEqual([1, 4096]); // 16*16*16 = 4096

            // Add next connection
            flatten.addNext(outputTensor);

            // Verify next connection
            expect(flatten.nexts[0]).toBe(outputTensor);

            // Delete connections
            flatten.deleteNext();
            flatten.deletePrev();

            // Verify disconnections
            expect(flatten.prevs[0]).toBeNull();
            expect(flatten.nexts[0]).toBeNull();
        });

        test('should reject incompatible input shapes', () => {
            // Create nodes with a high start_dim
            const flatten = new Op('00000000-0000-4000-8000-000000000001', 'Flatten', {
                start_dim: 3,
                end_dim: -1
            });
            
            // Tensor with insufficient dimensions
            const insufficientDimTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16], 'input');
            
            // Attempt to add prev connection with insufficient dimensions
            expect(() => {
                flatten.addPrev(insufficientDimTensor, [1, 16]);
            }).toThrow();
        });
    });

    describe('Conv3D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const conv3d = new Op('00000000-0000-4000-8000-000000000001', 'Conv3D', {
                in_channels: 3,
                out_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 3, 16, 32, 32], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 16, 32, 32], 'output');
            
            // Add prev connection
            conv3d.addPrev(inputTensor, [1, 3, 16, 32, 32]);
            
            // Verify prev connection
            expect(conv3d.prevs[0]).toBe(inputTensor);
            expect(conv3d.inShapes[0]).toEqual([1, 3, 16, 32, 32]);
            
            // Verify output shape updated
            expect(conv3d.outShapes[0]).toEqual([1, 16, 16, 32, 32]);
            
            // Add next connection
            conv3d.addNext(outputTensor);
            
            // Verify next connection
            expect(conv3d.nexts[0]).toBe(outputTensor);
            
            // Delete connections
            conv3d.deleteNext();
            conv3d.deletePrev();
            
            // Verify disconnections
            expect(conv3d.prevs[0]).toBeNull();
            expect(conv3d.nexts[0]).toBeNull();
        });
        
        test('should reject incompatible input shapes', () => {
            // Create nodes
            const conv3d = new Op('00000000-0000-4000-8000-000000000001', 'Conv3D', {
                in_channels: 3,
                out_channels: 16,
                kernel_size: 3,
                stride: 1,
                padding: 1
            });
            
            // Tensor with wrong number of channels
            const incompatibleTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 4, 16, 32, 32], 'input');
            
            // Attempt to add prev connection with wrong channel count
            expect(() => {
                conv3d.addPrev(incompatibleTensor, [1, 4, 16, 32, 32]);
            }).toThrow();
            
            // Tensor with wrong dimensionality
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 3, 32, 32], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                conv3d.addPrev(wrongDimTensor, [1, 3, 32, 32]);
            }).toThrow();
        });
    });

    describe('ConvTranspose2D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const convTranspose = new Op('00000000-0000-4000-8000-000000000001', 'ConvTranspose2D', {
                in_channels: 16,
                out_channels: 3,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                output_padding: 1
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16, 16, 16], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 3, 32, 32], 'output');
            
            // Add prev connection
            convTranspose.addPrev(inputTensor, [1, 16, 16, 16]);
            
            // Verify prev connection
            expect(convTranspose.prevs[0]).toBe(inputTensor);
            expect(convTranspose.inShapes[0]).toEqual([1, 16, 16, 16]);
            
            // Verify output shape updated (upsampled dimensions)
            expect(convTranspose.outShapes[0]).toEqual([1, 3, 32, 32]);
            
            // Add next connection
            convTranspose.addNext(outputTensor);
            
            // Verify next connection
            expect(convTranspose.nexts[0]).toBe(outputTensor);
            
            // Delete connections
            convTranspose.deleteNext();
            convTranspose.deletePrev();
            
            // Verify disconnections
            expect(convTranspose.prevs[0]).toBeNull();
            expect(convTranspose.nexts[0]).toBeNull();
        });
        
        test('should reject incompatible input shapes', () => {
            // Create nodes
            const convTranspose = new Op('00000000-0000-4000-8000-000000000001', 'ConvTranspose2D', {
                in_channels: 16,
                out_channels: 3,
                kernel_size: 3,
                stride: 2,
                padding: 1
            });
            
            // Tensor with wrong number of channels
            const incompatibleTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 8, 16, 16], 'input');
            
            // Attempt to add prev connection with wrong channel count
            expect(() => {
                convTranspose.addPrev(incompatibleTensor, [1, 8, 16, 16]);
            }).toThrow();
            
            // Tensor with wrong dimensionality
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 16], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                convTranspose.addPrev(wrongDimTensor, [1, 16, 16]);
            }).toThrow();
        });
    });

    describe('AvgPool2D Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const avgPool = new Op('00000000-0000-4000-8000-000000000001', 'AvgPool2D', {
                kernel_size: 2,
                stride: 2
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [1, 16, 32, 32], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 16, 16], 'output');
            
            // Add prev connection
            avgPool.addPrev(inputTensor, [1, 16, 32, 32]);
            
            // Verify prev connection
            expect(avgPool.prevs[0]).toBe(inputTensor);
            expect(avgPool.inShapes[0]).toEqual([1, 16, 32, 32]);
            
            // Verify output shape updated (halved dimensions)
            expect(avgPool.outShapes[0]).toEqual([1, 16, 16, 16]);
            
            // Add next connection
            avgPool.addNext(outputTensor);
            
            // Verify next connection
            expect(avgPool.nexts[0]).toBe(outputTensor);
            
            // Delete connections
            avgPool.deleteNext();
            avgPool.deletePrev();
            
            // Verify disconnections
            expect(avgPool.prevs[0]).toBeNull();
            expect(avgPool.nexts[0]).toBeNull();
        });
        
        test('should reject incompatible input shapes', () => {
            // Create nodes
            const avgPool = new Op('00000000-0000-4000-8000-000000000001', 'AvgPool2D', {
                kernel_size: 2,
                stride: 2
            });
            
            // Tensor with wrong dimensionality
            const wrongDimTensor = new Tensor('00000000-0000-4000-8000-000000000002', [16, 32], 'input');
            
            // Attempt to add prev connection with wrong dimensions
            expect(() => {
                avgPool.addPrev(wrongDimTensor, [16, 32]);
            }).toThrow();
            
            // Tensor with dimension too small for pooling
            const smallDimTensor = new Tensor('00000000-0000-4000-8000-000000000003', [1, 16, 1, 1], 'input');
            
            // Attempt to add prev connection with dimensions too small for the kernel size
            expect(() => {
                avgPool.addPrev(smallDimTensor, [1, 16, 1, 1]);
            }).toThrow();
        });
    });

    describe('Dropout Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const dropout = new Op('00000000-0000-4000-8000-000000000001', 'Dropout', {
                p: 0.5
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [32, 100], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [32, 100], 'output');
            
            // Add prev connection
            dropout.addPrev(inputTensor, [32, 100]);
            
            // Verify prev connection
            expect(dropout.prevs[0]).toBe(inputTensor);
            expect(dropout.inShapes[0]).toEqual([32, 100]);
            
            // Verify output shape unchanged (dropout preserves shape)
            expect(dropout.outShapes[0]).toEqual([32, 100]);
            
            // Add next connection
            dropout.addNext(outputTensor);
            
            // Verify next connection
            expect(dropout.nexts[0]).toBe(outputTensor);
            
            // Delete connections
            dropout.deleteNext();
            dropout.deletePrev();
            
            // Verify disconnections
            expect(dropout.prevs[0]).toBeNull();
            expect(dropout.nexts[0]).toBeNull();
        });
    });

    describe('Reshape Op', () => {
        test('should properly add and delete connections', () => {
            // Create nodes
            const reshape = new Op('00000000-0000-4000-8000-000000000001', 'Reshape', {
                shape: [32, 2, 50]
            });
            const inputTensor = new Tensor('00000000-0000-4000-8000-000000000002', [32, 100], 'input');
            const outputTensor = new Tensor('00000000-0000-4000-8000-000000000003', [32, 2, 50], 'output');
            
            // Add prev connection
            reshape.addPrev(inputTensor, [32, 100]);
            
            // Verify prev connection
            expect(reshape.prevs[0]).toBe(inputTensor);
            expect(reshape.inShapes[0]).toEqual([32, 100]);
            
            // Verify output shape updated according to reshape params
            expect(reshape.outShapes[0]).toEqual([32, 2, 50]);
            
            // Add next connection
            reshape.addNext(outputTensor);
            
            // Verify next connection
            expect(reshape.nexts[0]).toBe(outputTensor);
            
            // Delete connections
            reshape.deleteNext();
            reshape.deletePrev();
            
            // Verify disconnections
            expect(reshape.prevs[0]).toBeNull();
            expect(reshape.nexts[0]).toBeNull();
        });
    });
});
