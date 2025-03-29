import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group } from '@mantine/core';
import type { CheckerNodeType } from './checker';
import { LayerHandle } from './LayerHandle';
import { useGraphStore } from './store';

interface LayerBoxProps extends NodeProps {
    data: {
        type: CheckerNodeType;
        inputError?: string;
        outputError?: string;
    };
}

export function LayerBox({ data, id }: LayerBoxProps) {
    const { type, inputError, outputError } = data;
    const checkerNode = useGraphStore(state => state.checkerGraph.getNode(id));
    const inShape = checkerNode?.in_shape || undefined;
    const outShape = checkerNode?.out_shape || undefined;

    const card = (
        <Card shadow="sm" radius="md" withBorder style={{ width: 180, minHeight: 100 }}>
            <Card.Section withBorder inheritPadding py="xs">
                <Group justify="space-between">
                    <Text fw={500}>{type}</Text>
                </Group>
            </Card.Section>
        </Card>
    );

    return (
        <>
            <LayerHandle 
                position={Position.Left}
                type="target"
                dimensions={inShape}
                error={inputError}
            />

            {card}

            <LayerHandle 
                position={Position.Right}
                type="source"
                dimensions={outShape}
                error={outputError}
            />
        </>
    );
}
