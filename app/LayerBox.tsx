import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group } from '@mantine/core';
import { LayerHandle } from './LayerHandle';
import { useStore } from './store';
import { assert } from './utils';

interface LayerBoxProps extends NodeProps {
    data: {
        type: string;
        opType?: string;
        inputError?: string;
        outputError?: string;
    };
}

export function LayerBox({ data, id }: LayerBoxProps) {
    const { type, opType, inputError, outputError } = data;
    const node = useStore(state => state.checkerGraph?.getNode(id));
    assert(node, `No node in the graph with ID ${id}?? Something is very wrong`);

    const inShapes = node!.inShapes;
    const outShapes = node!.outShapes;

    const card = (
        <Card shadow="sm" radius="sm" withBorder style={{ padding: "8px 16px 8px" }}>
            <Group>
                <Text fw={500}>{opType || type}</Text>
            </Group>
        </Card>
    );

    // Render input handles based on node type
    const renderInputHandles = () => {
        // No shapes defined yet
        if (!inShapes.length) return (
            <LayerHandle
                position={Position.Left}
                type="target"
                id="0"
                dimensions={undefined}
                error={inputError}
            />
        );

        // Render a handle for each input shape
        return inShapes.map((shape, idx) => (
            <LayerHandle
                key={`in-${idx}`}
                position={Position.Left}
                type="target"
                dimensions={shape === null ? undefined : shape}
                error={inputError}
                id={idx.toString()}
                offset={idx}
                total={inShapes.length}
            />
        ));
    };

    // Render output handles based on node type
    const renderOutputHandles = () => {
        // No shapes defined yet
        if (!outShapes.length) return (
            <LayerHandle
                position={Position.Right}
                type="source"
                id="0"
                dimensions={undefined}
                error={outputError}
            />
        );

        // Render a handle for each output shape
        return outShapes.map((shape, idx) => (
            <LayerHandle
                key={`out-${idx}`}
                position={Position.Right}
                type="source"
                dimensions={shape === null ? undefined : shape}
                error={outputError}
                id={idx.toString()}
                offset={idx}
                total={outShapes.length}
            />
        ));
    };

    return (
        <>
            {renderInputHandles()}
            {card}
            {renderOutputHandles()}
        </>
    );
}
