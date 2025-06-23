import { execSync } from 'child_process';
import { CodeGenerator } from '../OpCompiler/codegen';
import { Graph } from '../OpCompiler/graph';
import { Tensor } from '../OpCompiler/tensor';
import { Linear } from '../moduledb/linear';
import { Op } from '../OpCompiler/op';
import path from 'path';

/**
 * Helper function to execute Python test template
 */
function testPyTorchModule(moduleCode: string, inputShapes: number[][]): any {
    const templatePath = path.join(__dirname, 'pytorch_test_template.py');
    const inputShapesJson = JSON.stringify(inputShapes);
    
    try {
        const result = execSync(
            `python3 "${templatePath}" "${moduleCode.replace(/"/g, '\\"')}" '${inputShapesJson}'`,
            { encoding: 'utf8', timeout: 10000 }
        );
        return JSON.parse(result);
    } catch (error: any) {
        // If Python script fails, still try to parse the output
        try {
            return JSON.parse(error.stdout || '{}');
        } catch {
            return {
                compilation_success: false,
                forward_success: false,
                error_message: `Python execution failed: ${error.message}`
            };
        }
    }
}

/**
 * Helper function to create a simple graph with one Linear layer
 */
function createLinearGraph(inputFeatures: number, outputFeatures: number): Graph {
    const graph = new Graph();
    
    // Create input tensor
    const inputTensor = new Tensor('input', [32, inputFeatures], 'input');
    
    // Create linear layer
    const linearParams = {
        input_features: inputFeatures,
        output_features: outputFeatures,
        bias: true
    };
    const linearOp = new Op('linear', 'Linear', linearParams);
    
    // Create output tensor
    const outputTensor = new Tensor('output', [32, outputFeatures], 'output');
    
    // Add nodes to graph
    graph.addNode('input', 'Tensor', { shape: [32, inputFeatures], variableName: 'input' });
    graph.addNode('linear', 'Op', { opType: 'Linear', ...linearParams });
    graph.addNode('output', 'Tensor', { shape: [32, outputFeatures], variableName: 'output' });
    
    // Connect nodes
    graph.connect('input', 'linear', 0, 0);
    graph.connect('linear', 'output', 0, 0);
    
    return graph;
}

describe('Linear Module Tests', () => {
    describe('Linear Layer', () => {
        test('should generate valid PyTorch code and execute successfully', () => {
            // Create a simple graph with Linear layer
            const graph = createLinearGraph(128, 64);
            const codeGen = new CodeGenerator(graph);
            
            // Generate PyTorch module code
            const moduleCode = codeGen.emitTorchModule();
            
            // Test with randomized input tensors
            const inputShapes = [[32, 128]]; // batch_size=32, input_features=128
            const result = testPyTorchModule(moduleCode, inputShapes);
            
            // Assertions
            expect(result.compilation_success).toBe(true);
            expect(result.forward_success).toBe(true);
            expect(result.output_shapes).toEqual([[32, 64]]); // batch_size=32, output_features=64
            
            if (!result.compilation_success) {
                console.error('Compilation failed:', result.error_message);
            }
            if (!result.forward_success) {
                console.error('Forward pass failed:', result.error_message);
            }
        });

        test('should handle different input/output dimensions', () => {
            const graph = createLinearGraph(784, 256);
            const codeGen = new CodeGenerator(graph);
            const moduleCode = codeGen.emitTorchModule();
            
            const inputShapes = [[16, 784]]; // Different batch size
            const result = testPyTorchModule(moduleCode, inputShapes);
            
            expect(result.compilation_success).toBe(true);
            expect(result.forward_success).toBe(true);
            expect(result.output_shapes).toEqual([[16, 256]]);
        });
    });

    describe('Linear ModuleDef', () => {
        test('should generate correct PyTorch module string', () => {
            const params = { input_features: 128, output_features: 64, bias: true };
            const moduleString = Linear.emitPytorchModule(params);
            
            expect(moduleString).toBe('nn.Linear(128, 64, bias=true)');
        });

        test('should infer correct output shape', () => {
            const inputShape = [32, 128];
            const params = { input_features: 128, output_features: 64 };
            const outputShape = Linear.inferOutputShape!(inputShape, params);
            
            expect(outputShape).toEqual([32, 64]);
        });

        test('should validate input shapes correctly', () => {
            const validShape = [32, 128];
            const invalidShape = [32, 256]; // Wrong input features
            const params = { input_features: 128, output_features: 64 };
            
            const validErrors = Linear.validateInputShape!(validShape, params);
            const invalidErrors = Linear.validateInputShape!(invalidShape, params);
            
            expect(validErrors).toEqual([]);
            expect(invalidErrors.length).toBeGreaterThan(0);
        });
    });
}); 