import React, { useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  BackgroundVariant,
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type { Layer } from './types';
import { LayerBox } from './LayerBox';

const nodeTypes = {
  default: LayerBox
};

const GRID_SIZE = 20;

export interface WorkspaceHandle {
  addLayer: (layer: Layer) => void;
}

export const Workspace = forwardRef<WorkspaceHandle>((_, ref) => {
  const [layers, setLayers] = useState<Layer[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nextX, setNextX] = useState(0);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );
  const onConnect: OnConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [],
  );

  const addLayer = useCallback((layer: Layer) => {
    const flowNode = {
      id: layer.id,
      type: 'default',
      data: layer,
      position: { x: nextX, y: 0 },
    } satisfies Node;

    setNextX(x => x + GRID_SIZE * 3); // Move next node 3 grid spaces to the right
    setLayers(l => [...l, layer]);
    setNodes(n => [...n, flowNode]);
  }, [nextX]);

  useImperativeHandle(ref, () => ({
    addLayer
  }));

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        snapToGrid={true}
        snapGrid={[GRID_SIZE, GRID_SIZE]}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={GRID_SIZE} size={1}/>
        <Controls />
      </ReactFlow>
    </div>
  );
});

Workspace.displayName = 'Workspace'; 