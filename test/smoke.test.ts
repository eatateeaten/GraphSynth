import { Graph } from '../OpCompiler/graph';
import { CodeGenerator } from '../OpCompiler/codegen';

describe('Smoke Tests for Refactored Graph and CodeGenerator', () => {

    test('simple tensor->tensor connection should generate code', () => {
        // Create graph
        const graph = new Graph();
        
        // Add nodes
        graph.addNode("input", "Tensor", { shape: [2, 3], variableName: "x" });
        graph.addNode("output", "Tensor", { shape: [2, 3], variableName: "y" });
        
        // Connect nodes
        graph.connect("input", "output", 0, 0);
        
        // Generate code (should not throw)
        const codeGen = new CodeGenerator(graph);
        const torchCode = codeGen.emitTorchFunctional();
        
        // Basic checks - should contain variable assignments and return
        expect(torchCode).toContain('v0 = x');
        expect(torchCode).toContain('return');
        expect(torchCode.length).toBeGreaterThan(0);
    });

    test('tensor->relu->tensor should generate code', () => {
        // Create graph
        const graph = new Graph();
        
        // Add nodes
        graph.addNode("input", "Tensor", { shape: [2, 3], variableName: "x" });
        graph.addNode("relu", "Op", { operation: "relu" });
        graph.addNode("output", "Tensor", { shape: [2, 3], variableName: "result" });
        
        // Connect nodes
        graph.connect("input", "relu", 0, 0);
        graph.connect("relu", "output", 0, 0);
        
        // Generate code (should not throw)
        const codeGen = new CodeGenerator(graph);
        const torchCode = codeGen.emitTorchFunctional();
        
        // Basic checks - should contain operations and return
        expect(torchCode).toContain('v0 = x');
        expect(torchCode).toContain('relu');
        expect(torchCode).toContain('return');
        expect(torchCode.length).toBeGreaterThan(0);
        
        console.log('Generated PyTorch code:');
        console.log(torchCode);
    });

    test('graph traversal methods work', () => {
        // Create graph
        const graph = new Graph();
        
        // Add nodes
        graph.addNode("input", "Tensor", { shape: [2, 3], variableName: "x" });
        graph.addNode("relu", "Op", { operation: "relu" });
        graph.addNode("output", "Tensor", { shape: [2, 3], variableName: "result" });
        
        // Connect nodes
        graph.connect("input", "relu", 0, 0);
        graph.connect("relu", "output", 0, 0);
        
        // Test BFS traversal
        const visitedNodes: string[] = [];
        graph.traverseBFS(node => {
            visitedNodes.push(node.id);
        });
        
        expect(visitedNodes).toContain('input');
        expect(visitedNodes).toContain('relu');
        expect(visitedNodes).toContain('output');
        expect(visitedNodes.length).toBe(3);
        
        // Test topological order
        const topoOrder = graph.getTopologicalOrder();
        expect(topoOrder.length).toBe(3);
        
        // Input should come before relu, relu before output
        const inputIndex = topoOrder.findIndex(n => n.id === 'input');
        const reluIndex = topoOrder.findIndex(n => n.id === 'relu');
        const outputIndex = topoOrder.findIndex(n => n.id === 'output');
        
        expect(inputIndex).toBeLessThan(reluIndex);
        expect(reluIndex).toBeLessThan(outputIndex);
    });

    test('sources and sinks computed correctly', () => {
        // Create graph
        const graph = new Graph();
        
        // Add nodes
        graph.addNode("input", "Tensor", { shape: [2, 3], variableName: "x" });
        graph.addNode("relu", "Op", { operation: "relu" });
        graph.addNode("output", "Tensor", { shape: [2, 3], variableName: "result" });
        
        // Connect nodes
        graph.connect("input", "relu", 0, 0);
        graph.connect("relu", "output", 0, 0);
        
        // Check sources and sinks
        const sources = graph.getSources();
        const sinks = graph.getSinks();
        
        expect(sources.size).toBe(1);
        expect(sinks.size).toBe(1);
        
        const sourceIds = Array.from(sources).map(n => n.id);
        const sinkIds = Array.from(sinks).map(n => n.id);
        
        expect(sourceIds).toContain('input');
        expect(sinkIds).toContain('output');
    });
});
