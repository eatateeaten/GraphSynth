import { Box, Code, Text } from '@mantine/core';
import { useStore } from './store';
import { useEffect, useState, useCallback } from 'react';

export function CodeWindow() {
    // Subscribe to nodes and edges instead of checkerGraph directly
    const nodes = useStore(state => state.nodes);
    const edges = useStore(state => state.edges);
    const checkerGraph = useStore(state => state.checkerGraph);

    // Use local state to store the generated code
    const [code, setCode] = useState("// No valid graph to generate code");

    // Function to format raw code into a PyTorch module
    const getFormattedCode = useCallback(() => {
        try {
            // Only generate code if there's a valid graph
            if (checkerGraph.getSources().size === 0 || checkerGraph.getSinks().size === 0) {
                return "// No valid graph to generate code";
            }
            
            // Get raw code
            const rawCode = checkerGraph.to_torch_functional();
            
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
            
            // Format code as a PyTorch module
            const formattedCode = `from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        
    def forward(self, ${sourceVars.join(', ')}):
${lines.filter(line => !line.includes('return')).map(line => '        ' + line).join('\n')}
        return ${sinkVars.join(', ')}
`;
            
            return formattedCode;
        } catch (e) {
            return `// Error generating code: ${e instanceof Error ? e.message : String(e)}`;
        }
    }, [checkerGraph]);

    // Generate code whenever nodes or edges change
    useEffect(() => {
        setCode(getFormattedCode());
    }, [nodes, edges, getFormattedCode]);

    return (
        <Box p="md" style={{ width: '100%', maxWidth: '100%' }}>
            <Text size="sm">Generated Code</Text>
            <Code block style={{"overflowWrap": "anywhere"}}>
                {code}
            </Code>
        </Box>
    );
}
