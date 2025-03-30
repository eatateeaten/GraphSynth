import { CheckerGraph, CheckerError } from '../app/checker';
import { v4 as uuidv4 } from 'uuid';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('CheckerGraph Advanced Tests', () => {
    let checker: CheckerGraph;

    beforeEach(() => {
        checker = new CheckerGraph();
    });

    test('Shape Inference During Connection', () => {
        // Create a tensor node with shape [1, 28, 28]
        const tensorId = uuidv4();
        checker.addNode({
            id: tensorId,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        // Create an op node (Conv2D) with null shape
        const convId = uuidv4();
        checker.addNode({
            id: convId,
            type: 'op',
            opType: 'Conv2d',
            params: {
                in_channels: 1,
                out_channels: 16,
                kernel_size: 3,
                padding: 1,
                stride: 1
            }
        });
        
        // Connect them
        const edgeResult = checker.connect(tensorId, convId);
        
        // Verify connection was successful
        expect(edgeResult).not.toBeInstanceOf(CheckerError);
        
        // Verify that Conv op is in ZophGraph with inferred input shape
        expect(checker.getNode(convId)).toBeDefined();
        
        // Create a tensor with incompatible shape [2, 28, 28] (batch size mismatch)
        const incompatibleTensorId = uuidv4();
        checker.addNode({
            id: incompatibleTensorId,
            type: 'tensor',
            params: { shape: [2, 28, 28] }
        });
        
        // Try connecting incompatible tensor to another Conv op
        const conv2Id = uuidv4();
        checker.addNode({
            id: conv2Id,
            type: 'op',
            opType: 'Conv2d',
            params: {
                in_channels: 16, // This expects 16 channels but tensor has 1
                out_channels: 32,
                kernel_size: 3
            }
        });
        
        // This should fail or result in a warning due to shape mismatch
        const badConnectionResult = checker.connect(incompatibleTensorId, conv2Id);
        
        // Either the connection failed with CheckerError or the node is not in ZophGraph
        if (!(badConnectionResult instanceof CheckerError)) {
            // If the edge was created in our wrapper but failed in ZophGraph,
            // the Conv2 node wouldn't be in the ZophGraph
            expect(checker.getNode(conv2Id)).toBeUndefined();
        }
    });

    test('Handling Reconnections After Parameter Changes', () => {
        // Create a chain: tensor -> op1 -> op2 -> op3
        const tensorId = uuidv4();
        const op1Id = uuidv4();
        const op2Id = uuidv4();
        const op3Id = uuidv4();
        
        // Add nodes
        checker.addNode({
            id: tensorId,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        checker.addNode({
            id: op1Id,
            type: 'op',
            opType: 'Conv2d',
            params: { 
                in_channels: 1,
                out_channels: 16,
                kernel_size: 3,
                padding: 1
            }
        });
        
        checker.addNode({
            id: op2Id,
            type: 'op',
            opType: 'ReLU',
            params: {}
        });
        
        checker.addNode({
            id: op3Id,
            type: 'op',
            opType: 'MaxPool2d',
            params: { kernel_size: 2, stride: 2 }
        });
        
        // Create connections
        checker.connect(tensorId, op1Id);
        checker.connect(op1Id, op2Id);
        checker.connect(op2Id, op3Id);
        
        // Edit op2's parameters (which should disconnect it)
        checker.editNodeParams(op2Id, { inplace: true });
        
        // Verify op2 is not in ZophGraph
        expect(checker.getNode(op2Id)).toBeUndefined();
        
        // Create a new connection from op1 to op3
        const bypassConnection = checker.connect(op1Id, op3Id);
        expect(bypassConnection).not.toBeInstanceOf(CheckerError);
        
        // Try to reconnect op2 back into the chain
        const resultToOp2 = checker.connect(op1Id, op2Id);
        expect(resultToOp2).not.toBeInstanceOf(CheckerError);
        
        const resultFromOp2 = checker.connect(op2Id, op3Id);
        expect(resultFromOp2).not.toBeInstanceOf(CheckerError);
        
        // Verify all nodes are now in ZophGraph
        expect(checker.getNode(op1Id)).toBeDefined();
        expect(checker.getNode(op2Id)).toBeDefined();
        expect(checker.getNode(op3Id)).toBeDefined();
    });

    test('Partial Graph Construction with Missing Shapes', () => {
        // Create several op nodes without connecting them to tensors
        const op1Id = uuidv4();
        const op2Id = uuidv4();
        const op3Id = uuidv4();
        
        // Add op nodes
        checker.addNode({
            id: op1Id,
            type: 'op',
            opType: 'Conv2d',
            params: { 
                in_channels: 1, 
                out_channels: 16, 
                kernel_size: 3 
            }
        });
        
        checker.addNode({
            id: op2Id,
            type: 'op',
            opType: 'ReLU',
            params: {}
        });
        
        checker.addNode({
            id: op3Id,
            type: 'op',
            opType: 'MaxPool2d',
            params: { kernel_size: 2 }
        });
        
        // Create a tensor node
        const tensorId = uuidv4();
        checker.addNode({
            id: tensorId,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        // Build connections in non-standard order
        // First connect op2 and op3
        const edge1 = checker.connect(op2Id, op3Id);
        
        // Then connect op1 to op2
        const edge2 = checker.connect(op1Id, op2Id);
        
        // Finally, connect the tensor at the beginning
        const edge3 = checker.connect(tensorId, op1Id);
        
        // All connections should exist at this point
        expect(edge1).not.toBeInstanceOf(CheckerError);
        expect(edge2).not.toBeInstanceOf(CheckerError);
        expect(edge3).not.toBeInstanceOf(CheckerError);
        
        // All nodes should be in ZophGraph now with properly inferred shapes
        expect(checker.getNode(tensorId)).toBeDefined();
        expect(checker.getNode(op1Id)).toBeDefined();
        expect(checker.getNode(op2Id)).toBeDefined();
        expect(checker.getNode(op3Id)).toBeDefined();
        
        // Create another op that would cause a shape conflict
        const incompatibleOpId = uuidv4();
        checker.addNode({
            id: incompatibleOpId,
            type: 'op',
            opType: 'Conv2d',
            params: { 
                in_channels: 3,  // Incompatible with tensor's 1 channel
                out_channels: 16, 
                kernel_size: 3 
            }
        });
        
        // Try to connect the tensor to this incompatible op
        const badConnection = checker.connect(tensorId, incompatibleOpId);
        
        // Verify this either returns an error or the node isn't added to ZophGraph
        if (!(badConnection instanceof CheckerError)) {
            // If we get here, check that the node isn't in ZophGraph
            expect(checker.getNode(incompatibleOpId)).toBeUndefined();
        }
    });
});
