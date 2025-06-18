import { Box, Code, Text, Button, Group } from '@mantine/core';
import { useStore } from './store';
import { useEffect, useState, useCallback } from 'react';

export function CodeWindow() {
    // Subscribe to nodes and edges instead of checkerGraph directly
    const nodes = useStore(state => state.nodes);
    const edges = useStore(state => state.edges);
    const checkerGraph = useStore(state => state.compilerGraph);
    const executeCodeInJupyter = useStore(state => state.executeCodeInJupyter);
    const isConnected = useStore(state => state.jupyter.status?.connected || false);
    const isExecuting = useStore(state => state.jupyter.isExecuting);

    // Use local state to store the generated code
    const [code, setCode] = useState("// No valid graph to generate code");
    const [showOutput, setShowOutput] = useState(false);

    // Estimate input tensor shapes based on the graph
    const getInputShapes = useCallback(() => {
        try {
            const sources = Array.from(checkerGraph.getSources());
            const shapes = sources.map(source => {
                // Try to get shape information from the graph node
                try {
                    const node = checkerGraph.getNode(source.id);
                    if (node && node.outShapes && node.outShapes.length > 0 && node.outShapes[0]) {
                        return node.outShapes[0];
                    }
                } catch (e) {
                    console.warn("Couldn't get shape for node:", source.id);
                }
                
                // Default shape if we can't determine it
                return [1, 3, 224, 224]; // Typical image input shape (batch, channels, height, width)
            });
            
            return shapes;
        } catch (e) {
            console.warn("Error getting input shapes:", e);
            return [];
        }
    }, [checkerGraph]);

    // Function to format raw code into a PyTorch module
    const getFormattedCode = useCallback(() => {
        try {
            // Only generate code if there's a valid graph
            if (checkerGraph.getSources().size === 0 || checkerGraph.getSinks().size === 0) {
                return "// No valid graph to generate code";
            }

            // Get raw code
            const rawCode = checkerGraph.emitTorchFunctional();

            // Get source and sink variables
            const sources = Array.from(checkerGraph.getSources());
            const sinks = Array.from(checkerGraph.getSinks());

            // Extract variable names from the raw code - this assumes the raw code uses specific patterns
            // Look for input variables (usually the first assignments or placeholders)
            let sourceVars: string[] = [];
            let lines = rawCode.split('\n');

            // Find lines with source variable references
            // Usually, these are the first variables being assigned
            let isFirstAssignment = true;
            for (const line of lines) {
                if (line.includes('return')) continue;
                if (line.trim() === '') continue;

                // Look for assignments where a variable appears on the right side without being defined first
                if (isFirstAssignment && line.includes('=')) {
                    // This is likely an input assignment (e.g., var1 = input_tensor)
                    const match = line.match(/(\w+)\s*=/);
                    if (match && match[1]) {
                        sourceVars.push(match[1]);
                        isFirstAssignment = false;
                    }
                }
            }

            // If we couldn't find source vars in code, use placeholders based on the number of sources
            if (sourceVars.length === 0) {
                sourceVars = Array(sources.length).fill(0).map((_, i) => `input_${i}`);
            }

            // Find return statements to extract sink variables
            const returnLine = lines.find(line => line.includes('return'));
            let sinkVars: string[] = [];

            if (returnLine) {
                // Extract variables from return statement
                const returnVars = returnLine.replace('return', '').trim();
                if (returnVars.includes(',')) {
                    // Multiple return values
                    sinkVars = returnVars.split(',').map(v => v.trim().replace(/[()]/g, ''));
                } else {
                    // Single return value
                    sinkVars = [returnVars];
                }
            } else {
                // If no return statement found, use the last variable assignment
                for (let i = lines.length - 1; i >= 0; i--) {
                    const line = lines[i];
                    if (line.includes('=') && !line.includes('return')) {
                        const match = line.match(/(\w+)\s+=\s+/);
                        if (match && match[1]) {
                            sinkVars = [match[1]];
                            break;
                        }
                    }
                }
            }
            
            // If we still couldn't find sink vars, use placeholders
            if (sinkVars.length === 0) {
                sinkVars = sinks.map((_, i) => `output_${i}`);
            }
            
            // Try to get input shapes for test data
            const inputShapes = getInputShapes();
            
            // Generate shape strings for input tensors
            const shapeStrings = inputShapes.map(shape => 
                shape && shape.length > 0 ? `[${shape.join(', ')}]` : '[1, 3, 224, 224]'
            );
            
            // Format code as a PyTorch module with test code
            const formattedCode = `import torch
import torch.nn as nn
import numpy as np

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        
    def forward(self, ${sourceVars.join(', ')}):
${lines.filter(line => !line.includes('return')).map(line => '        ' + line).join('\n')}
        return ${sinkVars.join(', ')}

# Create an instance of the model
model = MyModule()

# Generate random input tensors
${sourceVars.map((v, i) => `${v} = torch.randn(${shapeStrings[i] || '[1, 3, 224, 224]'})`).join('\n')}

# Run the model
print("Testing model with random inputs...")
with torch.no_grad():
    outputs = model(${sourceVars.join(', ')})

if isinstance(outputs, tuple):
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
else:
    print(f"Output shape: {outputs.shape}")
`;
            
            return formattedCode;
        } catch (e) {
            return `// Error generating code: ${e instanceof Error ? e.message : String(e)}`;
        }
    }, [checkerGraph, getInputShapes]);

    // Generate code whenever nodes or edges change
    useEffect(() => {
        setCode(getFormattedCode());
    }, [nodes, edges, getFormattedCode]);

    const handleExecuteCode = async () => {
        if (!isConnected) {
            alert("Please connect to a Jupyter server first.");
            return;
        }

        setShowOutput(true);
        await executeCodeInJupyter(code);
    };

    // AI: please stop trying to add CodeOutput here. it doesn't belong here
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
