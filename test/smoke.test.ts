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
        const torchCode = codeGen.emitTorchModule();
        
        // Basic checks - should contain module class definition
        expect(torchCode.length).toBeGreaterThan(0);
    });

    test('tensor->relu->tensor should generate code', () => {
        // Create graph
        const graph = new Graph();

        // Add nodes
        graph.addNode("input", "Tensor", { shape: [2, 3], variableName: "x" });
        graph.addNode("relu", "Op", { opType: "ReLU" });
        graph.addNode("output", "Tensor", { shape: [2, 3], variableName: "result" });

        // Connect nodes
        graph.connect("input", "relu", 0, 0);
        graph.connect("relu", "output", 0, 0);

        // Generate code (should not throw)
        const codeGen = new CodeGenerator(graph);
        const torchCode = codeGen.emitTorchModule();
        
        // Basic checks - should contain module structure
        expect(torchCode.length).toBeGreaterThan(0);
        
        console.log('Generated PyTorch module:');
        console.log(torchCode);
    });

    test('module generation creates proper PyTorch structure', () => {
        // Create a more complex graph
        const graph = new Graph();
        
        // Add nodes
        graph.addNode("input", "Tensor", { shape: [1, 3, 32, 32], variableName: "x" });
        graph.addNode("conv", "Op", { opType: "Conv2D", in_channels: 3, out_channels: 64, kernel_size: 3 });
        graph.addNode("relu", "Op", { opType: "ReLU" });
        graph.addNode("output", "Tensor", { shape: [1, 64, 30, 30], variableName: "out" });
        
        // Connect nodes
        graph.connect("input", "conv", 0, 0);
        graph.connect("conv", "relu", 0, 0);
        graph.connect("relu", "output", 0, 0);
        
        // Generate module
        const codeGen = new CodeGenerator(graph);
        const torchCode = codeGen.emitTorchModule();
        
        console.log('Complex module structure:');
        console.log(torchCode);
    });
});
