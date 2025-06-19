import { Box, Code, Text, Button, Group } from '@mantine/core';
import { useStore } from './store';
import { useEffect, useState, useCallback } from 'react';
import { CodeGenerator } from '../OpCompiler/codegen';

export function CodeWindow() {
    // Subscribe to nodes and edges instead of compilerGraph directly
    const nodes = useStore(state => state.nodes);
    const edges = useStore(state => state.edges);
    const compilerGraph = useStore(state => state.compilerGraph);
    const executeCodeInJupyter = useStore(state => state.executeCodeInJupyter);
    const isConnected = useStore(state => state.jupyter.status?.connected || false);
    const isExecuting = useStore(state => state.jupyter.isExecuting);

    // Use local state to store the generated code
    const [code, setCode] = useState("// No valid graph to generate code");

    // Generate PyTorch module code using the new CodeGenerator
    const getFormattedCode = useCallback(() => {
        try {
            // Only generate code if there's a valid graph
            if (compilerGraph.getSources().size === 0 || compilerGraph.getSinks().size === 0) {
                return "// No valid graph to generate code";
            }

            // Use the new CodeGenerator to generate a complete PyTorch module
            const codeGen = new CodeGenerator(compilerGraph);
            const moduleCode = codeGen.emitTorchModule();
            
            // Add test code to create and run the model
            const testCode = `

# Create an instance of the model
model = GeneratedModule()

# Generate random input tensors based on source nodes
import torch
${generateTestInputs()}

# Run the model
print("Testing model with random inputs...")
with torch.no_grad():
    outputs = model(${getInputVariableNames()})

if isinstance(outputs, tuple):
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
else:
    print(f"Output shape: {outputs.shape}")
`;
            
            return moduleCode + testCode;
        } catch (e) {
            return `# Error generating code: ${e instanceof Error ? e.message : String(e)}`;
        }
    }, [compilerGraph]);

    // Generate test input tensor creation code
    const generateTestInputs = useCallback(() => {
        try {
            const sources = Array.from(compilerGraph.getSources());
            return sources.map((source, i) => {
                // Try to get shape from the source node
                let shape = [1, 3, 224, 224]; // Default shape
                try {
                    const node = compilerGraph.getNode(source.id);
                    if (node && node.outShapes && node.outShapes.length > 0 && node.outShapes[0]) {
                        shape = node.outShapes[0];
                    }
                } catch (e) {
                    console.warn("Couldn't get shape for source node:", source.id);
                }
                
                const inputVar = sources.length === 1 ? "x" : `x${i}`;
                return `${inputVar} = torch.randn([${shape.join(', ')}])`;
            }).join('\n');
        } catch (e) {
            return "x = torch.randn([1, 3, 224, 224])  # Default input";
        }
    }, [compilerGraph]);

    // Get input variable names for the forward call
    const getInputVariableNames = useCallback(() => {
        try {
            const sources = Array.from(compilerGraph.getSources());
            if (sources.length === 1) {
                return "x";
            }
            return sources.map((_, i) => `x${i}`).join(', ');
        } catch (e) {
            return "x";
        }
    }, [compilerGraph]);

    // Generate code whenever nodes or edges change
    useEffect(() => {
        setCode(getFormattedCode());
    }, [nodes, edges, getFormattedCode]);

    const handleExecuteCode = async () => {
        if (!isConnected) {
            alert("Please connect to a Jupyter server first.");
            return;
        }

        await executeCodeInJupyter(code);
    };

    return (
        <Box style={{ width: '100%', maxWidth: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box p="md" style={{ display: 'flex', flexDirection: 'column' }}>
                <Group justify="space-between" mb="xs">
                    <Text size="sm">Generated Code</Text>
                    <Button 
                        size="xs" 
                        onClick={handleExecuteCode} 
                        disabled={!isConnected || code.includes("// No valid graph")}
                        loading={isExecuting}
                    >
                        Execute
                    </Button>
                </Group>
                <Code block style={{ overflowWrap: "anywhere", flexGrow: 1 }}>
                    {code}
                </Code>
            </Box>
        </Box>
    );
}
