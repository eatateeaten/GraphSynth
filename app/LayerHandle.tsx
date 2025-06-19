import { Handle, Position, type HandleProps } from 'reactflow';
import { Text, Stack, Popover, Box } from '@mantine/core';

// Define Shape type directly here
type Shape = number[];

interface LayerHandleProps extends Omit<HandleProps, 'position'> {
    dimensions?: Shape;
    position: Position;
    error?: string;
    id: string;
    offset?: number;
    total?: number; // Total number of handles for proper spacing
}

export function LayerHandle({ dimensions, position, error, type, offset = 0, total = 1, ...handleProps }: LayerHandleProps) {
    const content = (
        <div className="layer-handle-content">
            {dimensions ? (
                <Stack gap={4} align="center" style={{ width: '100%' }}>
                    {dimensions.map((dim: number, i: number) => (
                        <Box key={i} style={{
                            width: "100%",
                            minHeight: "20px",
                            border: "1px solid var(--mantine-color-dark-4)",
                            backgroundColor: 'var(--mantine-color-dark-6)'
                        }}>
                            <Text size="xs" color="cyan.2" style={{ width: '100%', textAlign: 'center' }}>
                                {dim}
                            </Text>
                        </Box>
                    ))}
                </Stack>
            ) : (
                <Box style={{
                    width: "100%",
                    minHeight: "20px",
                    border: "1px solid var(--mantine-color-dark-4)",
                    backgroundColor: 'var(--mantine-color-dark-6)'
                }}>
                    <Text size="xs" color="red.6" style={{ width: '100%', textAlign: 'center' }}>
                        ?
                    </Text>
                </Box>
            )}
        </div>
    );

    // Calculate vertical position based on offset and total
    const offsetPercent = total > 1 
        ? -10 + ((150 / (total - 1)) * offset) // Distribute handles evenly between 20% and 80%
        : 50; // Center if only one handle

    return (
        <Handle
            className="layer-handle"
            type={type}
            position={position}
            {...handleProps}
            style={{
                background: "none",
                borderColor: "none",
                border: "none",
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '4px 0',
                top: `${offsetPercent}%`, // Position handle vertically
                transform: 'translateY(-50%)', // Center the handle on its position
            }}
        >
            {error ? (
                <Popover
                    position="top"
                    withArrow
                    opened
                    shadow="md"
                    styles={{
                        dropdown: {
                            padding: '8px 12px',
                            maxWidth: '250px'
                        },
                        arrow: {
                        }
                    }}
                >
                    <Popover.Target>
                        {content}
                    </Popover.Target>
                    <Popover.Dropdown>
                        <Text size="sm" c="red.7" style={{ lineHeight: 1.4 }}>
                            {error}
                        </Text>
                    </Popover.Dropdown>
                </Popover>
            ) : content}
        </Handle>
    );
}
