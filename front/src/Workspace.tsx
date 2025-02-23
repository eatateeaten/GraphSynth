import React, { useCallback } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  BackgroundVariant,
  type Node,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type Edge,
  applyNodeChanges,
  applyEdgeChanges,
  Panel,
  type NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { LayerBox } from './LayerBox';
import { Button } from '@mantine/core';
import { useGraphStore } from './store';
import { GRID_SIZE } from './config';
import { wsManager } from './websocket';
import type { WSRequest } from './types';

// Generate 8-byte UUIDs for request IDs
const getRequestId = () => {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
};

// Define nodeTypes outside component to prevent recreation
const nodeTypes: NodeTypes = {
  default: LayerBox
};

export function Workspace() {
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const selectedNode = useGraphStore(state => state.selectedNode);
  const { updateNodes, updateEdges, setSelectedNode, deleteNode, addEdge, updateNodeData } = useGraphStore();

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => updateNodes(applyNodeChanges(changes, nodes)),
    [nodes, updateNodes]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => updateEdges(applyEdgeChanges(changes, edges)),
    [edges, updateEdges]
  );

  const handleConnect: OnConnect = useCallback(async (params) => {
    if (!params.source || !params.target) return;

    const sourceNode = nodes.find(n => n.id === params.source);
    const targetNode = nodes.find(n => n.id === params.target);
    if (!sourceNode || !targetNode) return;

    try {
      const sourceNodeId = params.source;
      const targetNodeId = params.target;

      // First add source node if it's still a bubble
      let sourceServerId = sourceNode.data.serverId;
      if (sourceNode.data.status === 'bubble') {
        updateNodeData(sourceNodeId, { status: 'validating' });
        const request: WSRequest = {
          requestId: getRequestId(),
          operation: 'addNode',
          type: sourceNode.data.type,
          params: sourceNode.data.params
        };
        const response = await wsManager.sendRequest(request);
        if (!response.success) throw new Error(response.error);
        sourceServerId = response.id;
        
        // Update source node with server response
        updateNodeData(sourceNodeId, {
          serverId: sourceServerId,
          status: 'valid',
          inShape: response.in_shape,
          outShape: response.out_shape,
          isValid: response.completed
        });
        console.log('Source node updated with:', { sourceServerId, response });
      }

      // Then add target node if it's still a bubble
      let targetServerId = targetNode.data.serverId;
      if (targetNode.data.status === 'bubble') {
        updateNodeData(targetNodeId, { status: 'validating' });
        const request: WSRequest = {
          requestId: getRequestId(),
          operation: 'addNode',
          type: targetNode.data.type,
          params: targetNode.data.params
        };
        const response = await wsManager.sendRequest(request);
        if (!response.success) throw new Error(response.error);
        targetServerId = response.id;
        
        // Update target node with server response
        updateNodeData(targetNodeId, {
          serverId: targetServerId,
          status: 'valid',
          inShape: response.in_shape,
          outShape: response.out_shape,
          isValid: response.completed
        });
      }

      // Now connect them using server IDs
      updateNodeData(targetNodeId, { status: 'validating' });

      const request: WSRequest = {
        requestId: getRequestId(),
        operation: 'setInputNode',
        nodeId: targetServerId,
        inputId: sourceServerId
      };
      const response = await wsManager.sendRequest(request);
      if (!response.success) throw new Error(response.error);

      // Add the edge using frontend IDs
      const newEdge: Edge = {
        id: `${sourceNode.id}-${targetNode.id}`,
        source: sourceNode.id,
        target: targetNode.id,
        sourceHandle: params.sourceHandle || undefined,
        targetHandle: params.targetHandle || undefined,
        type: 'default'
      };
      addEdge(newEdge);

      // Update target node with new shapes after connection
      updateNodeData(targetNodeId, {
        status: 'valid',
        inShape: response.in_shape,
        outShape: response.out_shape,
        isValid: response.completed
      });

    } catch (error: any) {
      console.error('Failed to connect nodes:', error);
      // Update nodes to show error state
      updateNodeData(params.source, {
        status: 'error',
        errorMessage: error.message
      });
      updateNodeData(params.target, {
        status: 'error',
        errorMessage: error.message
      });
    }
  }, [nodes, addEdge, updateNodeData]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
  }, [setSelectedNode]);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  const handleDelete = useCallback(() => {
    if (selectedNode) deleteNode(selectedNode);
  }, [selectedNode, deleteNode]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={handleConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        snapToGrid={true}
        snapGrid={[GRID_SIZE, GRID_SIZE]}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={GRID_SIZE} size={1}/>
        <Controls />
        <Panel position="top-right" style={{ padding: '10px' }}>
          <Button
            color="red"
            onClick={handleDelete}
            disabled={!selectedNode}
            style={{ opacity: selectedNode ? 1 : 0.5 }}
          >
            Delete Selected Node
          </Button>
        </Panel>
      </ReactFlow>
    </div>
  );
}

Workspace.displayName = 'Workspace'; 