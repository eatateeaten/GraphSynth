import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group, ActionIcon } from '@mantine/core';
import { IconPencil, IconTrash } from '@tabler/icons-react';
import type { Layer } from './types';
import { LayerHandle } from './LayerHandle';

interface LayerBoxProps extends NodeProps {
  data: Layer;
}

function getDimensions(layer: Layer) {
  switch (layer.type) {
    case 'conv1d':
      return {
        input: [layer.params.batch_size, layer.params.in_channels, layer.params.input_size],
        output: [layer.params.batch_size, layer.params.out_channels, layer.params.input_size]
      };
    case 'linear':
      return {
        input: [layer.params.batch_size, layer.params.input_features],
        output: [layer.params.batch_size, layer.params.output_features]
      };
    default:
      return { input: undefined, output: undefined };
  }
}

export function LayerBox({ data }: LayerBoxProps) {
  const dimensions = getDimensions(data);

  return (
    <>
      <LayerHandle 
        dimensions={dimensions.input}
        position={Position.Left}
        type="target"
      />

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
          {Object.entries(data.params).map(([key, value]) => (
            <div key={key}>
              <Text size="xs" c="dimmed">{key}:</Text>
              <Text size="sm">{value}</Text>
            </div>
          ))}
        </Card.Section>
      </Card>

      <LayerHandle 
        dimensions={dimensions.output}
        position={Position.Right}
        type="source"
      />
    </>
  );
} 