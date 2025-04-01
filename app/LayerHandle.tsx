import { Handle, Position, type HandleProps } from 'reactflow';
import { Text, Stack, Popover } from '@mantine/core';

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
                <Stack gap={2} align="center" style={{ width: '100%' }}>
                    {dimensions.map((dim: number, i: number) => (
                        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <Text size="xs" style={{ width: '100%', textAlign: 'center' }}>
                                {dim}
                            </Text>
                            {i < dimensions.length - 1 && (
                                <Text size="xs" c="dimmed">×</Text>
                            )}
                        </div>
                    ))}
                </Stack>
            ) : (
                <Text size="xs" c="red" style={{ width: '100%', textAlign: 'center' }}>
                    Any
                </Text>
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
                background: dimensions ? 'var(--mantine-color-blue-0)' : 'var(--mantine-color-red-0)',
                borderColor: dimensions ? 'var(--mantine-color-blue-3)' : 'var(--mantine-color-red-3)',
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
                            backgroundColor: 'var(--mantine-color-red-0)',
                            border: '1px solid var(--mantine-color-red-3)',
                            maxWidth: '250px'
                        },
                        arrow: {
                            backgroundColor: 'var(--mantine-color-red-0)',
                            border: '1px solid var(--mantine-color-red-3)',
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
