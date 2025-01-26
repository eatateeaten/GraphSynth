import { Handle, Position, type HandleProps } from 'reactflow';
import { Text, Stack } from '@mantine/core';

interface LayerHandleProps extends Omit<HandleProps, 'position'> {
  dimensions: any[] | undefined;
  position: Position;
}

export function LayerHandle({ dimensions, position, type, ...handleProps }: LayerHandleProps) {
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
      }}
    >
      <div className="layer-handle-content">
        {dimensions ? (
          <Stack gap={2} align="center" style={{ width: '100%' }}>
            {dimensions.map((dim, i) => (
              <>
                <Text size="xs" style={{ width: '100%', textAlign: 'center' }}>
                  {dim}
                </Text>
                {i < dimensions.length - 1 && (
                  <Text size="xs" c="dimmed">×</Text>
                )}
              </>
            ))}
          </Stack>
        ) : (
          <Text size="xs" c="red" style={{ width: '100%', textAlign: 'center' }}>
            Any
          </Text>
        )}
      </div>
    </Handle>
  );
}
