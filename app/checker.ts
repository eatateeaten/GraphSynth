import { assert } from 'console';
import { Graph as ZophGraph,
        GraphNode as ZophNode,
        Tensor as ZophTensor,
        Op as ZophOp,
        BranchOp as ZophBranch,
        MergeOp as ZophMerge } from '../isomorphic/graph';
import { v4 as uuidv4 } from 'uuid';

/* track edge information */
interface EdgeInfo {
  sourceId: string;
  targetId: string;
  sourceIndex?: number;
  targetIndex?: number;
}

/**
 * information about a pending node
 * AI: talk about ZophGraph limitations here
 * */
interface PendingNode {
  id: string; /* node ID. this is shared between CheckerGraph, ZophGraph and FlowGraph */
  type: string; /* is it a Tensor, Op, Merge, or Branch? */
  opType?: string; /* sub-type inside op, merge or branch */
  params?: Record<string, any>; /* parameters for the node */
  inputConnections: Set<string>; /* IDs of nodes connected to inputs */
  inputPorts: number; /* Number of required input ports */
}

interface NodeConfig {
  type: string; /* is it a Tensor, Op, Merge, or Branch? */
  opType?: string; /* sub-type inside op, merge or branch */
  params?: Record<string, any>; /* parameters for the node */
}

// Zoph Graph wrapper
export class CheckerGraph {
  private graph: ZophGraph = new ZophGraph();

  // Track nodes with unknown shapes separately
  private pendingNodes: Map<string, PendingNode> = new Map();

  // Track edge ID to connection info
  private edgeMap: Map<string, EdgeInfo> = new Map();

  // Track node ID to node instance
  private nodeMap: Map<string, ZophNode> = new Map();
  
  /** Add a node to the graph */
  addNode(nodeConfig: NodeConfig): string {
    const nodeId = uuidv4();

    // Only ZophTensors can be directly added to the graph without connections
    if (nodeConfig.type === 'tensor') {
        // Create a ZophTensor
        assert(nodeConfig.params !== undefined, "Shape is required for tensors");
        assert(nodeConfig.params?.shape !== undefined, "Shape is required for tensors");

        try {
            const node = new ZophTensor(nodeId, nodeConfig.params.shape, "torch");
        } catch (error) {
            return Error("Failed to create tensor");
        }

        this.graph.addNode(node);
        // Add to graph
        this.nodeMap.set(nodeId, node);
      }
    } else {
      // Store non-tensor nodes as pending until all inputs are connected
      const inputPorts = this.determineInputPorts(nodeConfig);
      
      this.pendingNodes.set(nodeId, {
        id: nodeId,
        type: nodeConfig.type,
        opType: nodeConfig.opType,
        params: nodeConfig.params || {},
        inputConnections: new Set<string>(),
        inputPorts
      });
    }
    
    return nodeId;
  }
  
  /**
   * Connect two nodes in the graph
   */
  connect(sourceId: string, targetId: string, sourceIndex?: number, targetIndex?: number): string {
    const edgeId = uuidv4();
    
    // Store edge info regardless of whether we can connect in ZophGraph yet
    this.edgeMap.set(edgeId, {
      sourceId,
      targetId,
      sourceIndex,
      targetIndex
    });
    
    // Handle connections for pending nodes
    const targetPending = this.pendingNodes.get(targetId);
    if (targetPending) {
      targetPending.inputConnections.add(sourceId);
      
      // Check if all inputs are now connected
      if (targetPending.inputConnections.size >= targetPending.inputPorts) {
        // All inputs are connected, try to add to ZophGraph
        this.tryAddPendingToZophGraph(targetId);
      }
    }
    
    // Try to connect in ZophGraph if both nodes exist there
    const sourceNode = this.nodeMap.get(sourceId);
    const targetNode = this.nodeMap.get(targetId);
    
    if (sourceNode && targetNode) {
      try {
        this.graph.connect(sourceNode, targetNode, sourceIndex, targetIndex);
      } catch (error) {
        // Connection failed in ZophGraph, but we still return the edge ID
        console.error("Failed to connect in ZophGraph:", error);
      }
    }
    
    return edgeId;
  }
  
  /**
   * Try to add a pending node to ZophGraph if all inputs are connected to nodes in ZophGraph
   */
  private tryAddPendingToZophGraph(nodeId: string): void {
    const pendingNode = this.pendingNodes.get(nodeId);
    if (!pendingNode) return;
    
    // Check if all input connections are to nodes in ZophGraph
    const allInputsInZoph = Array.from(pendingNode.inputConnections).every(
      inputId => this.nodeMap.has(inputId)
    );
    
    if (!allInputsInZoph) return;
    
    // Attempt to create the ZophNode
    try {
      let node: ZophNode;
      
      if (pendingNode.type === 'op') {
        // We would need to get the actual input shape here from connected nodes
        // This is a simplified implementation for demonstration
        const sourceNode = this.nodeMap.get(Array.from(pendingNode.inputConnections)[0]);
        if (!sourceNode) return;
        
        const inputShape = this.getOutputShape(sourceNode);
        if (!inputShape) return;
        
        node = new ZophOp(
          nodeId,
          inputShape,
          pendingNode.target || 'torch',
          pendingNode.opType || 'generic',
          pendingNode.params || {}
        );
      } else {
        // For other node types, we'd need specific handling
        // For now, just skip them
        return;
      }
      
      // Add to ZophGraph
      this.graph.addNode(node);
      this.nodeMap.set(nodeId, node);
      
      // Connect to all input nodes in ZophGraph
      for (const inputId of pendingNode.inputConnections) {
        const inputNode = this.nodeMap.get(inputId);
        if (inputNode) {
          // We would need to get the proper source/target indices here
          // This is a simplified implementation
          try {
            this.graph.connect(inputNode, node);
          } catch (error) {
            console.error("Failed to connect in ZophGraph during pending node addition:", error);
          }
        }
      }
      
      // Remove from pending
      this.pendingNodes.delete(nodeId);
      
    } catch (error) {
      console.error("Failed to add pending node to ZophGraph:", error);
    }
  }
  
  /**
   * Determine the number of input ports for a node config
   */
  private determineInputPorts(nodeConfig: any): number {
    // In a real implementation, this would depend on the specific node type
    // For now, just return a default value
    if (nodeConfig.type === 'op') {
      return 1; // Most ops have 1 input
    } else if (nodeConfig.type === 'merge') {
      return nodeConfig.inputCount || 2; // Merge ops have multiple inputs
    }
    return 1; // Default
  }
  
  /**
   * Get the output shape of a ZophNode
   */
  private getOutputShape(node: ZophNode): number[] | undefined {
    if (node instanceof ZophTensor) {
      return node.outShape;
    } else if (node instanceof ZophOp) {
      return node.outShape;
    } else if (node instanceof ZophMerge) {
      return node.outShape;
    } else if (node instanceof ZophBranch) {
      // Branch ops have multiple outputs, this would need specific handling
      return undefined;
    }
    return undefined;
  }
  
  /**
   * Edit node parameters
   */
  editNodeParams(nodeId: string, params: Record<string, any>): void {
    // If node is in ZophGraph, we need to disconnect it first
    const node = this.nodeMap.get(nodeId);
    
    if (node) {
      // Disconnect from all connections
      node.disconnectSource();
      node.disconnectSink();
      
      // If it's an Op with parameters, we need special handling
      if (node instanceof ZophOp) {
        // In a real implementation, we'd update the Op's parameters here
        // For now, we'll just remove the node and add it to pending
        this.graph.removeNode(node);
        this.nodeMap.delete(nodeId);
        
        // Add to pending with updated params
        this.pendingNodes.set(nodeId, {
          id: nodeId,
          type: 'op',
          target: node.target,
          opType: node.opType,
          params: params,
          inputConnections: new Set<string>(),
          inputPorts: 1
        });
      }
    }
    
    // If node was in pending nodes, update its params
    if (this.pendingNodes.has(nodeId)) {
      const pendingNode = this.pendingNodes.get(nodeId)!;
      pendingNode.params = { ...pendingNode.params, ...params };
      this.pendingNodes.set(nodeId, pendingNode);
    }
  }
  
  /**
   * Delete a node by ID
   */
  deleteNode(nodeId: string): void {
    // Remove from ZophGraph if it exists there
    const node = this.nodeMap.get(nodeId);
    if (node) {
      this.graph.removeNode(node);
      this.nodeMap.delete(nodeId);
    }
    
    // Remove from pending nodes if it exists there
    this.pendingNodes.delete(nodeId);
    
    // Clean up any edges that reference this node
    for (const [edgeId, edgeInfo] of this.edgeMap.entries()) {
      if (edgeInfo.sourceId === nodeId || edgeInfo.targetId === nodeId) {
        this.edgeMap.delete(edgeId);
      }
    }
    
    // Update input connections for any pending nodes
    for (const [id, pendingNode] of this.pendingNodes.entries()) {
      if (pendingNode.inputConnections.has(nodeId)) {
        pendingNode.inputConnections.delete(nodeId);
        this.pendingNodes.set(id, pendingNode);
      }
    }
  }
  
  /**
   * Delete an edge by ID
   */
  deleteEdge(edgeId: string): void {
    // Get edge info
    const edgeInfo = this.edgeMap.get(edgeId);
    if (!edgeInfo) return;
    
    // Update input connections for target if it's a pending node
    const targetPending = this.pendingNodes.get(edgeInfo.targetId);
    if (targetPending) {
      targetPending.inputConnections.delete(edgeInfo.sourceId);
    }
    
    // Try to disconnect in ZophGraph if both nodes exist there
    const sourceNode = this.nodeMap.get(edgeInfo.sourceId);
    const targetNode = this.nodeMap.get(edgeInfo.targetId);
    
    if (sourceNode && targetNode) {
      try {
        this.graph.disconnect(sourceNode, targetNode, edgeInfo.sourceIndex, edgeInfo.targetIndex);
      } catch (error) {
        console.error("Failed to disconnect in ZophGraph:", error);
      }
    }
    
    // Remove edge from our map
    this.edgeMap.delete(edgeId);
  }
  
  /**
   * Get a node by ID
   */
  getNode(nodeId: string): ZophNode | undefined {
    return this.nodeMap.get(nodeId);
  }
}
