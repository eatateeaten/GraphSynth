import { Position, type NodeProps } from 'reactflow';
import { Card, Text, Group } from '@mantine/core';
import { LayerHandle } from './LayerHandle';
import { useStore } from './store';
import { PendingNode } from '../isomorphic/graph';

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
    let node = useStore(state => state.checkerGraph?.getNode(id) || state.checkerGraph?.getPendingNode(id));
    const pending = node instanceof PendingNode;

    // Get shapes, handling all possible types (null, number[], number[][])
    const inShape = node?.inShape;
    const outShape = node?.outShape;

    const card = (
        <Card shadow="sm" radius="sm" withBorder style={{ padding: "8px 16px 8px" }}>
            <Group>
                <Text fw={500}>{opType || type}</Text>
                {pending ? <Text c='rgba(194, 89, 89, 1)'>●</Text> : <></>}
            </Group>
        </Card>
    );

    // Render input handles based on node type
    const renderInputHandles = () => {
        // For null shape, render a single "Any" handle
        if (!inShape) return (
            <LayerHandle
                position={Position.Left}
                type="target"
                dimensions={undefined}
                error={inputError}
            />
        );

        // Array of arrays - multiple input shapes (MergeOp)
        if (Array.isArray(inShape[0])) {
            return (inShape as number[][]).map((shape, idx) => (
                <LayerHandle
                    key={`in-${idx}`}
                    position={Position.Left}
                    type="target"
                    dimensions={shape}
                    error={inputError}
                    handleId={idx.toString()}
                    offset={idx}
                />
            ));
        }

        // Simple array - single input shape (Tensor, Op, BranchOp)
        return (
            <LayerHandle
                position={Position.Left}
                type="target"
                dimensions={inShape as number[]}
                error={inputError}
            />
        );
    };

    // Render output handles based on node type
    const renderOutputHandles = () => {
        // For null shape, render a single "Any" handle
        if (!outShape) return (
            <LayerHandle
                position={Position.Right}
                type="source"
                dimensions={undefined}
                error={outputError}
            />
        );

        // Array of arrays - multiple output shapes (BranchOp)
        if (Array.isArray(outShape[0])) {
            return (outShape as number[][]).map((shape, idx) => (
                <LayerHandle
                    key={`out-${idx}`}
                    position={Position.Right}
                    type="source"
                    dimensions={shape}
                    error={outputError}
                    handleId={idx.toString()}
                    offset={idx}
                />
            ));
        }

        // Simple array - single output shape (Tensor, Op, MergeOp)
        return (
            <LayerHandle
                position={Position.Right}
                type="source"
                dimensions={outShape as number[]}
                error={outputError}
            />
        );
    };

    return (
        <>
            {renderInputHandles()}
            {card}
            {renderOutputHandles()}
        </>
    );
}
