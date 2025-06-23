import { Box, Text, Code, Loader } from '@mantine/core';
import { useStore } from './store';
import { ansiToHtml } from './utils';

export function CodeOutput() {
    const { lastExecutionResult, isExecuting } = useStore(state => state.jupyter);
    
    if (isExecuting) {
        return (
            <Box p="md" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Loader size="md" />
                <Text size="sm" mt="xs">Executing code...</Text>
            </Box>
        );
    }
    
    if (!lastExecutionResult) {
        return (
            <Box p="md">
                <Text size="sm" color="dimmed">No code has been executed yet.</Text>
                <Text size="xs" color="dimmed">Use the Execute button to run your code.</Text>
            </Box>
        );
    }
    
    const { success, textOutput } = lastExecutionResult;
    
    return (
        <Box p="md">
            <Text 
                size="sm" 
                fw={500} 
                mb="xs" 
                color={success ? 'teal' : 'red'}
            >
                {success ? 'Execution Successful' : 'Execution Failed'}
            </Text>
            
            <Code 
                block 
                style={{
                    overflow: 'auto',
                    whiteSpace: 'pre-wrap',
                    backgroundColor: '#000000'
                }}
            >
                <div dangerouslySetInnerHTML={{ __html: ansiToHtml(textOutput) || 'No output' }} />
            </Code>
        </Box>
    );
}
