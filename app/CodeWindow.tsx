import { Box, Text } from '@mantine/core';
import { useStore } from './store';
import { useEffect, useState } from 'react';

export function CodeWindow() {
    // Subscribe to nodes and edges instead of checkerGraph directly
    const nodes = useStore(state => state.nodes);
    const edges = useStore(state => state.edges);
    const checkerGraph = useStore(state => state.checkerGraph);

    // Use local state to store the generated code
    const [code, setCode] = useState("// No valid graph to generate code");

    // Generate code whenever nodes or edges change
    useEffect(() => {
        try {
            // Only generate code if there's a valid graph
            if (checkerGraph.getSources().size > 0 && checkerGraph.getSinks().size > 0) {
                setCode(checkerGraph.to_torch_functional());
            } else {
                setCode("// No valid graph to generate code");
            }
        } catch (e) {
            setCode(`// Error generating code: ${e instanceof Error ? e.message : String(e)}`);
        }
    }, [nodes, edges, checkerGraph]);

    return (
        <Box p="md" style={{ width: '100%', maxWidth: '100%' }}>
            <Text size="sm">Generated Code</Text>
            <pre style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '1rem',
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '0.9rem',
                lineHeight: 1.5,
                width: '100%',
                maxWidth: '100%',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word'
            }}>
                {code}
            </pre>
        </Box>
    );
}
