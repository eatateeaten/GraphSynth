import { Graph } from '../isomorphic/graph';
import { v4 as uuidv4 } from 'uuid';

describe('Branch Operations', () => {
    describe('Split Operation Shape Validation', () => {
        test('Valid split: equal sections along channel dimension', () => {
            const graph = new Graph();
            const inputId = uuidv4();
            const splitId = uuidv4();
            const output1Id = uuidv4();
            const output2Id = uuidv4();

            // Create nodes
            graph.createPendingNode('Tensor', inputId, {
                shape: [1, 64, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Split', splitId, {
                target: 'torch',
                inShape: [1, 64, 32, 32],
                splitParams: {
                    dim: 1,
                    sections: [32, 32]
                }
            });

            graph.createPendingNode('Tensor', output1Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Tensor', output2Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            // Make connections
            graph.makeTensorSource(inputId);
            graph.connect(inputId, splitId);
            graph.connect(splitId, output1Id, 0);
            graph.connect(splitId, output2Id, 1);

            // Should not throw
            expect(() => graph.validate_graph()).not.toThrow();
        });

        test('Invalid split: sections sum does not match input dimension', () => {
            const graph = new Graph();
            const inputId = uuidv4();
            const splitId = uuidv4();
            const output1Id = uuidv4();
            const output2Id = uuidv4();

            // Create nodes
            graph.createPendingNode('Tensor', inputId, {
                shape: [1, 64, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Split', splitId, {
                target: 'torch',
                inShape: [1, 64, 32, 32],
                splitParams: {
                    dim: 1,
                    sections: [32, 40]  // Sum is 72, but input has 64 channels
                }
            });

            graph.createPendingNode('Tensor', output1Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Tensor', output2Id, {
                shape: [1, 40, 32, 32],
                target: 'torch'
            });

            // Make connections
            graph.makeTensorSource(inputId);
            graph.connect(inputId, splitId);
            graph.connect(splitId, output1Id, 0);
            graph.connect(splitId, output2Id, 1);

            // Should throw
            expect(() => graph.validate_graph()).toThrow();
        });

        test('Invalid split: output shape mismatch', () => {
            const graph = new Graph();
            const inputId = uuidv4();
            const splitId = uuidv4();
            const output1Id = uuidv4();
            const output2Id = uuidv4();

            // Create nodes
            graph.createPendingNode('Tensor', inputId, {
                shape: [1, 64, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Split', splitId, {
                target: 'torch',
                inShape: [1, 64, 32, 32],
                splitParams: {
                    dim: 1,
                    sections: [32, 32]
                }
            });

            graph.createPendingNode('Tensor', output1Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Tensor', output2Id, {
                shape: [1, 32, 16, 16],  // Wrong shape
                target: 'torch'
            });

            // Make connections
            graph.makeTensorSource(inputId);
            graph.connect(inputId, splitId);
            graph.connect(splitId, output1Id, 0);
            graph.connect(splitId, output2Id, 1);

            // Should throw
            expect(() => graph.validate_graph()).toThrow();
        });

        test('Invalid split: invalid dimension', () => {
            const graph = new Graph();
            const inputId = uuidv4();
            const splitId = uuidv4();
            const output1Id = uuidv4();
            const output2Id = uuidv4();

            // Create nodes
            graph.createPendingNode('Tensor', inputId, {
                shape: [1, 64, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Split', splitId, {
                target: 'torch',
                inShape: [1, 64, 32, 32],
                splitParams: {
                    dim: 4,  // Invalid dimension (input only has 4 dimensions)
                    sections: [32, 32]
                }
            });

            graph.createPendingNode('Tensor', output1Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            graph.createPendingNode('Tensor', output2Id, {
                shape: [1, 32, 32, 32],
                target: 'torch'
            });

            // Make connections
            graph.makeTensorSource(inputId);
            graph.connect(inputId, splitId);
            graph.connect(splitId, output1Id, 0);
            graph.connect(splitId, output2Id, 1);

            // Should throw
            expect(() => graph.validate_graph()).toThrow();
        });
    });
}); 