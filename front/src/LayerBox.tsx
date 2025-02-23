import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group, ActionIcon } from '@mantine/core';
import { IconPencil, IconTrash } from '@tabler/icons-react';
import type { Layer } from './types';
import { LayerHandle } from './LayerHandle';

interface LayerBoxProps extends NodeProps {
  data: Layer;
}

function getDimensions(layer: Layer) {
  if (layer.sourceness === 'source') {
    // Source nodes only have output shape
    return {
      input: undefined,
      output: layer.params.shape || layer.outShape
    };
  }
  // For middle/sink nodes, use inShape/outShape
  return {
    input: layer.inShape,
    output: layer.outShape
  };
}

function renderParams(layer: Layer) {
  switch (layer.type) {
    case 'tensor':
      return layer.params.shape ? (
        <div>
          <Text size="xs" c="dimmed">shape:</Text>
          <Text size="sm">[{layer.params.shape.join(', ')}]</Text>
        </div>
      ) : null;
    case 'reshape':
      return layer.params.out_dim ? (
        <div>
          <Text size="xs" c="dimmed">out_dim:</Text>
          <Text size="sm">[{layer.params.out_dim.join(', ')}]</Text>
        </div>
      ) : null;
    default:
      return null;
  }
}

export function LayerBox({ data }: LayerBoxProps) {
  const dimensions = getDimensions(data);

  return (
    <>
      {/* Only show input handle for non-source nodes */}
      {data.sourceness !== 'source' && (
        <LayerHandle 
          dimensions={dimensions.input}
          position={Position.Left}
          type="target"
        />
      )}

      <Card shadow="sm" radius="md" withBorder style={{ width: 180, minHeight: 180 }}>
        <Card.Section withBorder inheritPadding py="xs">
          <Group justify="space-between">
            <Text fw={500}>{data.type}</Text>
            <Group gap="xs">
              <ActionIcon variant="subtle" color="gray" size="sm">
                <IconPencil size={14} />
              </ActionIcon>
              <ActionIcon variant="subtle" color="red" size="sm">
                <IconTrash size={14} />
              </ActionIcon>
            </Group>
          </Group>
        </Card.Section>

        <Card.Section p="sm">
          {renderParams(data)}
        </Card.Section>
      </Card>

      {/* Only show output handle for non-sink nodes */}
      {data.sourceness !== 'sink' && (
        <LayerHandle 
          dimensions={dimensions.output}
          position={Position.Right}
          type="source"
        />
      )}
    </>
  );
}
