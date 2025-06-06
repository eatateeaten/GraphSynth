import { Graph } from '../compiler/graph';
import { Tensor } from '../compiler/tensor';
import { Op } from '../compiler/op';

describe('Graph Operations', () => {
    test('should create and add nodes to graph', () => {
        const graph = new Graph();

        // Add a tensor node
        const tensorId = graph._generateUUID();
        graph.addNode(tensorId, 'Tensor', {
            shape: [3, 224, 224],
            variableName: 'input'
        });

        // Add an op node
        const opId = graph._generateUUID();
        graph.addNode(opId, 'Op', {
            opType: 'Conv2d',
            kernelSize: 3,
            stride: 1,
            padding: 1,
            inChannels: 3,
            outChannels: 16
        });

        // Verify nodes exist in graph
        const tensorNode = graph.getNode(tensorId);
        const opNode = graph.getNode(opId);

        expect(tensorNode).toBeDefined();
        expect(opNode).toBeDefined();
        expect(tensorNode).toBeInstanceOf(Tensor);
        expect(opNode).toBeInstanceOf(Op);
        expect(tensorNode?.id).toBe(tensorId);
        expect(opNode?.id).toBe(opId);
    });

    test('should connect and disconnect nodes', () => {
        const graph = new Graph();

        // Create nodes
        const tensorId = graph._generateUUID();
        graph.addNode(tensorId, 'Tensor', {
            shape: [1, 28, 28],
            variableName: 'input_tensor'
        });

        const reluId = graph._generateUUID();
        graph.addNode(reluId, 'Op', {
            opType: 'ReLU'
        });

        // Connect nodes
        graph.connect(tensorId, reluId, 0, 0);

        // Verify connection
        const tensorNode = graph.getNode(tensorId);
        const reluNode = graph.getNode(reluId);

        expect(tensorNode?.nexts[0]).toBe(reluNode);
        expect(reluNode?.prevs[0]).toBe(tensorNode);

        // Disconnect nodes
        graph.disconnect(tensorId, reluId, 0, 0);

        // Verify disconnection
        expect(tensorNode?.nexts[0]).toBeNull();
        expect(reluNode?.prevs[0]).toBeNull();
    });

    test('should delete nodes and clean up connections', () => {
        const graph = new Graph();

        // Create a simple network: tensor -> split -> [tensor1, tensor2]
        const inputId = graph._generateUUID();
        const splitId = graph._generateUUID();
        const output1Id = graph._generateUUID();
        const output2Id = graph._generateUUID();

        // Add nodes
        graph.addNode(inputId, 'Tensor', {
            shape: [10, 10],
            variableName: 'x'
        });

        graph.addNode(splitId, 'Split', {
            dim: 0,
            sections: [5, 5]
        });

        graph.addNode(output1Id, 'Tensor', {
            shape: [5, 10],
            variableName: 'y1'
        });

        graph.addNode(output2Id, 'Tensor', {
            shape: [5, 10],
            variableName: 'y2'
        });

        // Connect nodes
        graph.connect(inputId, splitId, 0, 0);
        graph.connect(splitId, output1Id, 0, 0);
        graph.connect(splitId, output2Id, 1, 0);

        // Delete split node
        graph.removeNode(splitId);

        // Verify split node is removed
        expect(graph.getNode(splitId)).toBeUndefined();

        // Verify connections are cleaned up
        const inputNode = graph.getNode(inputId);
        const output1Node = graph.getNode(output1Id);
        const output2Node = graph.getNode(output2Id);

        expect(inputNode?.nexts[0]).toBeNull();
        expect(output1Node?.prevs[0]).toBeNull();
        expect(output2Node?.prevs[0]).toBeNull();
    });
});
