import { CheckerGraph, CheckerError } from '../app/checker';
import { v4 as uuidv4 } from 'uuid';
import { describe, test, expect, beforeEach } from '@jest/globals';

describe('CheckerGraph Basic Tests', () => {
    let checker: CheckerGraph;

    beforeEach(() => {
        checker = new CheckerGraph();
    });

    test('Basic Node Addition and Connection', () => {
        // Create a tensor node with a valid shape
        const tensorId = uuidv4();
        const tensorResult = checker.addNode({
            id: tensorId,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });

        // Verify tensor was added successfully
        expect(tensorResult).toBe(tensorId);
        
        // Create an op node (with null shape)
        const opId = uuidv4();
        const opResult = checker.addNode({
            id: opId,
            type: 'op',
            opType: 'ReLU',
            params: {}
        });

        // Verify op was added successfully
        expect(opResult).toBe(opId);
        
        // Connect the tensor to the op
        const edgeResult = checker.connect(tensorId, opId);
        
        // Verify the connection was successful
        expect(typeof edgeResult).toBe('string');
        expect(edgeResult).not.toBeInstanceOf(CheckerError);
        
        // Verify that both nodes are in the ZophGraph
        expect(checker.getNode(tensorId)).toBeDefined();
        expect(checker.getNode(opId)).toBeDefined();
    });

    test('Parameter Editing', () => {
        // Create an op node with initial parameters
        const opId = uuidv4();
        checker.addNode({
            id: opId,
            type: 'op',
            opType: 'Conv2d',
            params: { kernel_size: 3, padding: 1, stride: 1 }
        });
        
        // Create a tensor node and connect it to the op
        const tensorId = uuidv4();
        checker.addNode({
            id: tensorId,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        // Connect tensor to op
        checker.connect(tensorId, opId);
        
        // Edit the parameters
        const editResult = checker.editNodeParams(opId, { kernel_size: 5, padding: 2 });
        
        // Verify that editNodeParams didn't return an error
        expect(editResult).not.toBeInstanceOf(CheckerError);
        
        // The op node should now be in pendingNodes, not in ZophGraph
        expect(checker.getNode(opId)).toBeUndefined();
        
        // Reconnect op to tensor
        const reconnectResult = checker.connect(tensorId, opId);
        
        // Verify reconnection was successful
        expect(reconnectResult).not.toBeInstanceOf(CheckerError);
        
        // Op should be back in ZophGraph
        expect(checker.getNode(opId)).toBeDefined();
    });

    test('Node and Edge Deletion', () => {
        // Create a chain of nodes: tensor1 -> op1 -> op2 -> tensor2
        const tensor1Id = uuidv4();
        const op1Id = uuidv4();
        const op2Id = uuidv4();
        const tensor2Id = uuidv4();
        
        // Add nodes
        checker.addNode({
            id: tensor1Id,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        checker.addNode({
            id: op1Id,
            type: 'op',
            opType: 'Conv2d',
            params: { kernel_size: 3 }
        });
        
        checker.addNode({
            id: op2Id,
            type: 'op',
            opType: 'ReLU',
            params: {}
        });
        
        checker.addNode({
            id: tensor2Id,
            type: 'tensor',
            params: { shape: [1, 28, 28] }
        });
        
        // Connect nodes
        const edge1 = checker.connect(tensor1Id, op1Id);
        const edge2 = checker.connect(op1Id, op2Id);
        const edge3 = checker.connect(op2Id, tensor2Id);
        
        // Delete a node in the middle (op1)
        const deleteResult = checker.deleteNode(op1Id);
        
        // Verify deletion was successful
        expect(deleteResult).not.toBeInstanceOf(CheckerError);
        
        // Verify the node is removed
        expect(checker.getNode(op1Id)).toBeUndefined();
        
        // Delete an edge and verify it's removed
        const edgeId = edge3 as string;
        const edgeDeleteResult = checker.deleteEdge(edgeId);
        
        // Verify edge deletion was successful
        expect(edgeDeleteResult).not.toBeInstanceOf(CheckerError);
    });
});
