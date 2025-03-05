import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group } from '@mantine/core';
import type { CheckerNodeType } from './checker';
import { LayerHandle } from './LayerHandle';
import { useGraphStore } from './store';

interface LayerBoxProps extends NodeProps {
  data: {
    type: CheckerNodeType;
    errorMessage?: string;
  };
}

export function LayerBox({ data, id }: LayerBoxProps) {
  const { type, errorMessage } = data;
  const checkerNode = useGraphStore(state => state.checkerGraph.getNode(id));
  const inShape = checkerNode?.in_shape || undefined;
  const outShape = checkerNode?.out_shape || undefined;

  return (
    <>
      <LayerHandle 
        position={Position.Left}
        type="target"
        dimensions={inShape}
      />

      <Card shadow="sm" radius="md" withBorder style={{ width: 180, minHeight: 100 }}>
        <Card.Section withBorder inheritPadding py="xs">
          <Group justify="space-between">
            <Text fw={500}>{type}</Text>
          </Group>
        </Card.Section>

        {errorMessage && (
          <Card.Section p="sm">
            <Text size="sm" c="red">{errorMessage}</Text>
          </Card.Section>
        )}
      </Card>

      <LayerHandle 
        position={Position.Right}
        type="source"
        dimensions={outShape}
      />
    </>
  );
}
