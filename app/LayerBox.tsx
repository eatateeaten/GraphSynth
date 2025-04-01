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
    
    // Use the node type to determine handle configuration
    const isBranchNode = type === 'Split' || type === 'Copy';
    const isMergeNode = type === 'Concat' || type === 'PointwiseReduce';

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
        // No shape or pending split/copy (they don't know their input shape yet)
        if (!inShape || (pending && isBranchNode)) return (
            <LayerHandle
                position={Position.Left}
                type="target"
                id="0"
                dimensions={undefined}
                error={inputError}
            />
        );

        // Multiple inputs (MergeOp)
        if (Array.isArray(inShape[0]) || isMergeNode) {
            const shapes = inShape as number[][];
            return shapes.map((shape, idx) => (
                <LayerHandle
                    key={`in-${idx}`}
                    position={Position.Left}
                    type="target"
                    dimensions={shape}
                    error={inputError}
                    id={idx.toString()}
                    offset={idx}
                    total={shapes.length}
                />
            ));
        }

        // Single input shape (Tensor, Op, BranchOp)
        return (
            <LayerHandle
                position={Position.Left}
                type="target"
                id="0"
                dimensions={inShape as number[]}
                error={inputError}
            />
        );
    };

    // Render output handles based on node type
    const renderOutputHandles = () => {
        // No shape
        if (!outShape) return (
            <LayerHandle
                position={Position.Right}
                type="source"
                id="0"
                dimensions={undefined}
                error={outputError}
            />
        );

        // Multiple outputs (BranchOp)
        if (Array.isArray(outShape[0]) || isBranchNode) {
            const shapes = outShape as number[][];
            return shapes.map((shape, idx) => (
                <LayerHandle
                    key={`out-${idx}`}
                    position={Position.Right}
                    type="source"
                    dimensions={shape}
                    error={outputError}
                    id={idx.toString()}
                    offset={idx}
                    total={shapes.length}
                />
            ));
        }

        // Single output shape (Tensor, Op, MergeOp)
        return (
            <LayerHandle
                position={Position.Right}
                type="source"
                id="0"
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
