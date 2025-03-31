import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp } from './branch_op';
import { MergeOp, Concat } from './merge_op';
import { Split } from './branch_op';

export { Tensor, Op, Concat, Split, BranchOp, MergeOp };

/**
 * A wrapper class for nodes that are not yet connected of the main graph. This is our way to maintain that all members of _nodes will be connected 
 * It delegates all GraphNode methods to the wrapped node.
 */
export class PendingNode<T extends GraphNode> extends GraphNode {
    private _wrappedNode: T;

    constructor(node: T) {
        super(node.id, node.target);
        this._wrappedNode = node;
    }
    
    /**
     * Factory method to create a PendingNode with a new GraphNode of the specified type
     * @param type Type of GraphNode to create ("Tensor", "Op", "Split", "Concat", etc.)
     * @param id UUID for the node
     * @param target Target framework ("torch", "jax", etc.)
     * @param params Additional parameters required for node construction
     * @returns A PendingNode wrapping the created GraphNode
     */
    static create(type: string, id: string, target: string, params: Record<string, any> = {}): PendingNode<GraphNode> {
        let node: GraphNode;

        switch (type) {
            case "Tensor":
                if (!params.shape) {
                    throw new Error("Shape parameter is required for Tensor");
                }
                node = new Tensor(id, params.shape, target, params.variableName || null);
                break;
                
            case "Op":
                if (!params.opType) {
                    throw new Error("opType parameter is required for Op");
                }
                node = new Op(id, target, params.opType, params.opParams || {});
                break;
                
            case "Split":
                if (!params.inShape) {
                    throw new Error("inShape parameter is required for Split");
                }
                if (!params.splitParams || !params.splitParams.dim || !params.splitParams.sections) {
                    throw new Error("splitParams with dim and sections is required for Split");
                }
                node = new Split(id, params.inShape, target, params.splitParams);
                break;
                
            case "Concat":
                if (!params.inShapes) {
                    throw new Error("inShapes parameter is required for Concat");
                }
                if (!params.concatParams || params.concatParams.dim === undefined) {
                    throw new Error("concatParams with dim is required for Concat");
                }
                node = new Concat(id, params.inShapes, target, params.concatParams);
                break;
                
            default:
                throw new Error(`Unknown GraphNode type: ${type}`);
        }
        
        return new PendingNode(node);
    }
    // Delegating properties
    get prev(): GraphNode | null { return this._wrappedNode.prev; }
    set prev(node: GraphNode | null) { this._wrappedNode.prev = node; }
    get next(): GraphNode | null { return this._wrappedNode.next; }
    set next(node: GraphNode | null) { this._wrappedNode.next = node; }

    // Implement shape and parameter accessors
    get inShape(): number[] | null | number[][] { return this._wrappedNode.inShape; }
    get outShape(): number[] | null | number[][] { return this._wrappedNode.outShape; }
    get params(): Record<string, any> { return this._wrappedNode.params; }

    // Delegating methods
    addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void { this._wrappedNode.addPrev(prev, indexSelf, indexPrev); }
    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void { this._wrappedNode.addNext(next, indexSelf, indexNext); }
    deletePrev(indexSelf?: number): void { this._wrappedNode.deletePrev(indexSelf); }
    deleteNext(indexSelf?: number): void { this._wrappedNode.deleteNext(indexSelf); }
    to_torch_functional(inputs: string[]): string { return this._wrappedNode.to_torch_functional(inputs); }
    // Accessor for the wrapped node
    unwrap(): T { return this._wrappedNode; }
}

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _sources: Set<GraphNode>;
    private _sinks: Set<GraphNode>;
    private _pendingNodes: Map<string, PendingNode<GraphNode>>;  // Changed to Map for O(1) lookups by ID

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
        this._pendingNodes = new Map(); 
    }

    /**
     * Adds a node to the pending collection
     * @private
     */
    private _addPendingNode<T extends GraphNode>(node: T): PendingNode<T> {
        // Check if the node already exists in pending nodes or main graph
        if (this._pendingNodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} already exists in pending nodes`);
        }
        if (this._nodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} already exists in the graph`);
        }
        
        // Wrap and add node to pending nodes
        const pendingNode = new PendingNode<T>(node);
        this._pendingNodes.set(node.id, pendingNode as PendingNode<GraphNode>);
        return pendingNode;
    }

    /**
     * Creates a new node and adds it to the pending collection.
     * 
     * Pending nodes exist outside the main graph until they're connected to a node in the graph.
     * This is the recommended way to create nodes for later use in the graph.
     * 
     * @param type - Type of node to create ("Tensor", "Op", "Split", "Concat")
     * @param id - UUID for the node (must be in valid UUID v4 format)
     * @param params - Parameters required for node construction:
     *   - For "Tensor": { shape: number[], target?: string, variableName?: string }
     *   - For "Op": { opType: string, opParams?: Record<string, any>, target?: string }
     *   - For "Split": { inShape: number[], splitParams: { dim: number, sections: number[] }, target?: string }
     *   - For "Concat": { inShapes: number[][], concatParams: { dim: number }, target?: string }
     * @returns A PendingNode wrapping the created node
     * 
     * @example
     * // Create a pending tensor
     * const pendingTensor = graph.createPendingNode("Tensor", "550e8400-e29b-41d4-a716-446655440000", {
     *   shape: [1, 3, 224, 224],
     *   target: "torch",
     *   variableName: "input_image"  // Optional name to use in generated code
     * });
     * 
     * // Create a pending operation
     * const pendingOp = graph.createPendingNode("Op", "550e8400-e29b-41d4-a716-446655440001", {
     *   opType: "relu",
     *   target: "torch"
     * });
     */
    createPendingNode(type: string, id: string, params: Record<string, any> = {}): PendingNode<GraphNode> {
        // Check if the node already exists in pending nodes or main graph
        if (this._pendingNodes.has(id)) {
            throw new Error(`Node with id ${id} already exists in pending nodes`);
        }
        if (this._nodes.has(id)) {
            throw new Error(`Node with id ${id} already exists in the graph`);
        }
        
        // Create a pending node of the specified type
        const pendingNode = PendingNode.create(type, id, params.target || "torch", params);
        this._pendingNodes.set(id, pendingNode);
        return pendingNode;
    }
    
    /**
     * Removes a node from the pending collection.
     * 
     * This is useful for cleaning up pending nodes that are no longer needed.
     * Only applies to nodes in the pending state; connected nodes are part of the main graph.
     * 
     * @param nodeId - ID of the pending node to remove
     * @throws Error if the node doesn't exist in the pending collection
     * 
     * @example
     * // Remove a pending node that's no longer needed
     * graph.removePendingNode("550e8400-e29b-41d4-a716-446655440000");
     */
    removePendingNode(nodeId: string): void {
        if (!this._pendingNodes.has(nodeId)) {
            throw new Error(`Node with id ${nodeId} is not a pending node`);
        }
        
        this._pendingNodes.delete(nodeId);
    }

    /**
     * Promotes a pending Tensor node to a source node in the main graph.
     * 
     * This method allows adding an input tensor to the graph without requiring
     * it to be connected to another node first. It will move the node from
     * the pending collection to the main graph and mark it as a source.
     * 
     * @param nodeId - ID of the pending node to promote as a source
     * @throws Error if the node doesn't exist in the pending collection
     * @throws Error if the node is not a Tensor (only Tensors can be sources)
     * 
     * @example
     * // Create a pending tensor
     * const pendingTensor = graph.createPendingNode("Tensor", "input-tensor-id", {
     *   shape: [1, 3, 224, 224],
     *   target: "torch",
     *   variableName: "input_image"
     * });
     * 
     * // Promote it to a source node
     * graph.makeTensorSource("input-tensor-id");
     */
    makeTensorSource(nodeId: string): void {
        // Check if the node exists in pending nodes
        if (!this._pendingNodes.has(nodeId)) {
            throw new Error(`Node with id ${nodeId} is not a pending node`);
        }
        
        // Get the pending node and unwrap it
        const pendingNode = this._pendingNodes.get(nodeId)!;
        const node = pendingNode.unwrap();
        
        // Verify that the node is a Tensor
        if (!(node instanceof Tensor)) {
            throw new Error(`Cannot make node with id ${nodeId} a source: only Tensor nodes can be sources`);
        }
        
        // Remove from pending nodes and add to main graph
        this._pendingNodes.delete(nodeId);
        this._nodes.set(nodeId, node);
        
        // Add to sources
        this._sources.add(node);
    }

    /**
     * Removes a node from the main graph.
     * 
     * This method only works for nodes that have no active connections.
     * To remove a connected node, you must first disconnect all its connections.
     * 
     * @param nodeId - ID of the node to remove from the graph
     * @throws Error if the node doesn't exist or has active connections
     * 
     * @example
     * // Disconnect a node's connections first
     * graph.disconnect("sourceNodeId", "nodeToRemoveId");
     * 
     * // Then remove the node
     * graph.removeNode("nodeToRemoveId");
     */
    removeNode(nodeId: string): void {
        const node = this._nodes.get(nodeId);
        if (!node) {throw new Error(`Node with id ${nodeId} does not exist in graph`);}
        // Check if the node has any connections using a helper function
        const hasConnections = this._nodeHasConnections(node);
        if (hasConnections) {throw new Error(`Cannot remove node ${nodeId}: node has active connections`);}
        // Remove from collections
        this._nodes.delete(nodeId);
        this._sources.delete(node);
        this._sinks.delete(node);
    }

    private _nodeHasConnections(node: GraphNode): boolean {
        if (node instanceof Tensor || node instanceof Op || node instanceof BranchOp) {if (node.prev !== null) return true;}
        if (node instanceof Tensor || node instanceof Op || node instanceof MergeOp) {if (node.next !== null) return true; }
        if (node instanceof BranchOp && node._nexts.some(n => n !== null)) return true;
        if (node instanceof MergeOp && node._prevs.some(p => p !== null)) return true;
        return false;
    }

    /**
     * Connects two nodes in the graph.
     * 
     * The source node must be in the main graph, but the sink can be either in the main graph
     * or in the pending collection. If the sink is a pending node, it will be promoted to
     * the main graph after a successful connection.
     * 
     * @param sourceId - ID of the source node
     * @param sinkId - ID of the sink node
     * @param sourceIndex - Index for the source node output (required for BranchOp)
     * @param sinkIndex - Index for the sink node input (required for MergeOp)
     * @throws Error if nodes don't exist, indices are invalid, or shapes don't match
     * 
     * @example
     * // Connect a tensor to an operation
     * graph.connect("tensorId", "opId");
     * 
     * // Connect with specific indices (for branch/merge operations)
     * graph.connect("splitId", "opId", 0); // Connect to first output of split
     * graph.connect("opId", "concatId", undefined, 1); // Connect to second input of concat
     */
    connect(sourceId: string, sinkId: string, sourceIndex?: number, sinkIndex?: number): void {
        // Get source node from main graph
        const source = this._nodes.get(sourceId);
        if (!source) {throw new Error(`Source node with id ${sourceId} does not exist in graph`);}

        // Get sink node from either main graph or pending nodes
        let sink = this._nodes.get(sinkId) || this._pendingNodes.get(sinkId);
        if (!sink) {throw new Error(`Sink node with id ${sinkId} does not exist in graph or pending nodes`);}
        
        // Determine if sink is pending and unwrap it if needed
        const sinkIsPending = sink instanceof PendingNode;
        if (sinkIsPending) {
            sink = (sink as PendingNode<GraphNode>).unwrap();
        }
          
        // Validate connection endpoints and get shapes
        const sourceOutShape = this._validateSourceAndGetOutShape(source, sourceIndex);
        const sinkInShape = this._validateSinkAndGetInShape(sink, sinkIndex);

        // Skip shape compatibility check if sink has no defined input shape yet
        // This allows connecting nodes before their shapes are fully determined
        if (sinkInShape === null) {
            // Skip shape check - allow connection without shape validation
        } else if (!GraphNode.shapeMatch(sourceOutShape, sinkInShape)) {
            throw new Error(`Shape mismatch: Cannot connect ${source.constructor.name} with output shape [${sourceOutShape}] to ${sink.constructor.name} with input shape [${sinkInShape}]`);
        }
        // Establish bidirectional connections
        // Let each node handle its own connection logic
        sink.addPrev(source, sinkIndex, sourceIndex);
        source.addNext(sink, sourceIndex, sinkIndex);

        // Only if the connection was successful and sink was pending, move it to the main graph
        if (sinkIsPending) {
            this._pendingNodes.delete(sink.id);
            this._nodes.set(sink.id, sink);
        }
        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    /**
     * Validates the source node's connection point and returns its output shape
     */
    private _validateSourceAndGetOutShape(source: GraphNode, sourceIndex?: number): number[] {
        // For BranchOp, validate the index
        if (source instanceof BranchOp) {
            if (sourceIndex === undefined) {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
            sourceIndex = GraphNode.checkIndexInBound(sourceIndex, source.outShape.length, "connect (BranchOp output)");
            
            // Get specific output shape for the branch
            const branchOutShape = source.outShape[sourceIndex];
            if (branchOutShape === null || branchOutShape === undefined) {
                throw new Error(`Cannot connect from BranchOp with id ${source.id} at output ${sourceIndex}: output shape is undefined`);
            }
            return branchOutShape;
        } 
        
        // For other node types (Tensor, Op, MergeOp)
        if (source instanceof Tensor || source instanceof Op || source instanceof MergeOp) {
            if (source.outShape === null) {
                throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
            }
            return source.outShape;
        }
        // Unknown node type
        throw new Error(`Unknown source node type: ${source.constructor.name}`);
    }

    /**
     * Validates the sink node's connection point and returns its input shape, returned value can be Null 
     */
    private _validateSinkAndGetInShape(sink: GraphNode, sinkIndex?: number): number[] | null {
        // For MergeOp, validate the index and get specific input shape
        if (sink instanceof MergeOp) {
            if (sinkIndex === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            sinkIndex = GraphNode.checkIndexInBound(sinkIndex, sink.inShape.length, "connect (MergeOp input)");
            if (!sink.inShape || !sink.inShape[sinkIndex]) {
                throw new Error(`MergeOp with id ${sink.id} must have defined input shape at index ${sinkIndex}`);
            }
            return sink.inShape[sinkIndex];
        }
        
        // For Op and BranchOp, allow undefined input shape
        if (sink instanceof Op || sink instanceof BranchOp) {
            return sink.inShape || null;
        }
        
        // For Tensor, input shape must be defined
        if (sink instanceof Tensor) {
            if (!sink.inShape) {
                throw new Error(`Tensor with id ${sink.id} must have defined input shape`);
            }
            return sink.inShape;
        }
        
        // Unknown node type
        throw new Error(`Unknown sink node type: ${sink.constructor.name}`);
    }

    /**
     * Disconnects two nodes.
     * 
     * This method breaks the connection between two nodes in the main graph.
     * Both nodes must exist in the main graph (not in pending nodes).
     * After disconnection, shape validation may need to be performed again when reconnecting.
     * 
     * @param sourceId - ID of the source node
     * @param sinkId - ID of the sink node
     * @param sourceIndex - Index for the source node output (required for BranchOp)
     * @param sinkIndex - Index for the sink node input (required for MergeOp)
     * @throws Error if nodes don't exist or are not properly connected
     * 
     * @example
     * // Disconnect two nodes
     * graph.disconnect("opId", "tensorId");
     * 
     * // Disconnect with specific indices (for branch/merge operations)
     * graph.disconnect("splitId", "opId", 0); // Disconnect from first output of split
     * graph.disconnect("opId", "concatId", undefined, 1); // Disconnect from second input of concat
     */
    disconnect(sourceId: string, sinkId: string, sourceIndex?: number, sinkIndex?: number): void {
        // Get source and sink nodes from the main graph
        const source = this._nodes.get(sourceId);
        if (!source) {
            throw new Error(`Source node with id ${sourceId} does not exist in graph`);
        }
        
        const sink = this._nodes.get(sinkId);
        if (!sink) {
            throw new Error(`Sink node with id ${sinkId} does not exist in graph`);
        }
        
        if (source instanceof BranchOp) {
            if (sourceIndex === undefined) {
                throw new Error("When disconnecting from a BranchOp, an output index must be specified");
            }
            sourceIndex = GraphNode.checkIndexInBound(sourceIndex, source.outShape.length, "disconnect (BranchOp output)");
        }
        if (sink instanceof MergeOp) {
            if (sinkIndex === undefined) {
                throw new Error("When disconnecting to a MergeOp, an input index must be specified");
            }
            sinkIndex = GraphNode.checkIndexInBound(sinkIndex, sink.inShape.length, "disconnect (MergeOp input)");
        }
        // Verify connections exist
        this._verifyConnectionExists(source, sink, sourceIndex, sinkIndex);

        // Break connections using node methods
        sink.deletePrev(sinkIndex);
        source.deleteNext(sourceIndex);

        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    private _verifyConnectionExists(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        // Check source to sink connection
        let sourceHasConnection = false;
        if (source instanceof Tensor || source instanceof Op || source instanceof MergeOp) {
            sourceHasConnection = source.next === sink;
        } else if (source instanceof BranchOp) {
            if (sourceIndex === undefined) {
                throw new Error("When disconnecting from a BranchOp, an output index must be specified");
            }
            sourceHasConnection = source._nexts[sourceIndex] === sink;
        }

        // Check sink to source connection
        let sinkHasConnection = false;
        if (sink instanceof Tensor || sink instanceof Op || sink instanceof BranchOp) {
            sinkHasConnection = sink.prev === source;
        } else if (sink instanceof MergeOp) {
            if (sinkIndex === undefined) {
                throw new Error("When disconnecting to a MergeOp, an input index must be specified");
            }
            sinkHasConnection = sink._prevs[sinkIndex] === source;
        }

        // If connections don't match, issue a warning but proceed
        if (!sourceHasConnection || !sinkHasConnection) {
            console.warn(`Warning: Disconnecting nodes that may not be properly connected: ${source.id} -> ${sink.id}`);
        }
    }

    private _refreshNodeSinkSourceStatus(node: GraphNode): void {
        // Check source status (no incoming connections)
        const isSource = node instanceof MergeOp
            ? node._prevs.every(p => !p)
            : node.prev === null;
            
        // Check sink status (no outgoing connections)
        const isSink = node instanceof BranchOp
            ? node._nexts.every(n => !n)
            : node.next === null;

        // Update collections
        if (isSource) this._sources.add(node);
        else this._sources.delete(node);
        
        if (isSink) this._sinks.add(node);
        else this._sinks.delete(node);
    }

    getNode(id: string): GraphNode | undefined {
        return this._nodes.get(id);
    }

    getPendingNode(id: string): PendingNode<GraphNode> | undefined {
        return this._pendingNodes.get(id) as PendingNode<GraphNode> | undefined;
    }

    getSources(): Set<GraphNode> {
        return new Set(this._sources);
    }

    getSinks(): Set<GraphNode> {
        return new Set(this._sinks);
    }

    validate_graph(): void {
        // Check that the graph has source nodes
        if (this._sources.size === 0) {
            throw new Error("Graph has no source nodes");
        }
        
        // Check that the graph has sink nodes
        if (this._sinks.size === 0) {
            throw new Error("Graph has no sink nodes");
        }
        
        // Check that all source nodes are Tensors
        const sourceNodes = Array.from(this._sources);
        for (const source of sourceNodes) {
            if (!(source instanceof Tensor)) {
                throw new Error(`Source node ${source.id} is not a Tensor (found ${source.constructor.name} instead)`);
            }
        }
        
        // Check that all sink nodes are Tensors
        const sinkNodes = Array.from(this._sinks);
        for (const sink of sinkNodes) {
            if (!(sink instanceof Tensor)) {
                throw new Error(`Sink node ${sink.id} is not a Tensor (found ${sink.constructor.name} instead)`);
            }
        }
        
        // Check that all sinks are reachable from sources using BFS
        const visited = new Set<string>();
        const queue: GraphNode[] = Array.from(this._sources);
        
        // BFS traversal from source nodes
        while (queue.length > 0) {
            const node = queue.shift()!;
            
            if (visited.has(node.id)) {
                continue;
            }
            
            visited.add(node.id);
            
            // Add next nodes to the queue
            if (node instanceof Tensor || node instanceof Op) {
                if (node.next) {
                    queue.push(node.next);
                }
            } else if (node instanceof BranchOp) {
                for (const nextNode of (node as BranchOp)._nexts) {
                    if (nextNode) {
                        queue.push(nextNode);
                    }
                }
            } else if (node instanceof MergeOp) {
                if (node.next) {
                    queue.push(node.next);
                }
            }
        }
        
        // Check if any nodes are unreachable
        if (visited.size !== this._nodes.size) {
            const unreachable = Array.from(this._nodes.keys())
                .filter(id => !visited.has(id))
                .map(id => this._nodes.get(id)!.id);
            
            throw new Error(`Graph contains unreachable nodes: ${unreachable.join(', ')}`);
        }
        
        // Check if all sinks are reachable
        for (const sink of sinkNodes) {
            if (!visited.has(sink.id)) {
                throw new Error(`Sink node ${sink.id} is not reachable from any source`);
            }
        }
        
        // Check for cycles
        this._checkForCycles();
    }
    
    private _checkForCycles(): void {
        // Track nodes being processed in the current DFS path
        const visiting = new Set<string>();
        // Track nodes already fully processed
        const visited = new Set<string>();
        
        // Start DFS from each source node
        const sourceNodes = Array.from(this._sources);
        for (const source of sourceNodes) {
            this._dfsCheckCycle(source, visiting, visited);
        }
    }
    
    private _dfsCheckCycle(node: GraphNode, visiting: Set<string>, visited: Set<string>): void {
        // If already fully processed, no need to check again
        if (visited.has(node.id)) {
            return;
        }
        
        // If we're visiting this node in the current path, we found a cycle
        if (visiting.has(node.id)) {
            throw new Error(`Graph contains a cycle involving node ${node.id}`);
        }
        
        // Mark node as being visited in current path
        visiting.add(node.id);
        
        // Visit all next nodes
        if (node instanceof Tensor || node instanceof Op) {
            if (node.next) {
                this._dfsCheckCycle(node.next, visiting, visited);
            }
        } else if (node instanceof BranchOp) {
            for (const nextNode of (node as BranchOp)._nexts) {
                if (nextNode) {
                    this._dfsCheckCycle(nextNode, visiting, visited);
                }
            }
        } else if (node instanceof MergeOp) {
            if (node.next) {
                this._dfsCheckCycle(node.next, visiting, visited);
            }
        }
        
        // Mark node as fully processed
        visiting.delete(node.id);
        visited.add(node.id);
    }

    
    /**
     * Generates functional PyTorch code from the graph.
     * Uses a depth-first traversal strategy to respect computation order.
     * 
     * @returns A string containing PyTorch code in functional style
     */
    toTorchFunctional(): string {
        // Validate the graph
        this.validate_graph();
        
        // Initialize code string
        let code = "";
        
        // Create a variable counter for generating unique variable names
        const varCounter = { value: 0 };
        
        // Map to track multi-input nodes and their collected input variables
        const multiInputNodes = new Map<GraphNode, string[]>();
        
        // Set to track used variable names to ensure uniqueness
        const usedVarNames = new Set<string>();
        
        // Stack for DFS traversal - contains nodes and their input variables
        const stack: Array<{node: GraphNode, inputs: string[]}> = [];
        
        // Start with source nodes
        const sourceNodes = Array.from(this._sources);
        for (const sourceNode of sourceNodes) {
            if (sourceNode instanceof Tensor) {
                // Generate variable name for source tensor
                let varName = "";
                if (sourceNode.variableName) {
                    // If variableName is already used, append a number to make it unique
                    varName = sourceNode.variableName;
                    let counter = 1;
                    while (usedVarNames.has(varName)) {
                        varName = `${sourceNode.variableName}_${counter++}`;
                    }
                } else {
                    // Generate a unique variable name
                    do {
                        varName = `var_${varCounter.value++}`;
                    } while (usedVarNames.has(varName));
                }
                
                // Mark this variable name as used
                usedVarNames.add(varName);
                
                // Add variable declaration for input tensor
                code += `${varName} = ${sourceNode.to_torch_functional([], [varName])}\n`;
                
                // Add its next node to the stack
                if (sourceNode.next) {
                    stack.push({ node: sourceNode.next, inputs: [varName] });
                }
            }
        }
        
        // Process the stack until empty
        while (stack.length > 0) {
            const { node, inputs } = stack.pop()!;
            
            // Generate output variable name
            let outputVar;
            do {
                outputVar = `var_${varCounter.value++}`;
            } while (usedVarNames.has(outputVar));
            usedVarNames.add(outputVar);
            
            // Generate code for this node
            if (node instanceof BranchOp) {
                // Branch operations need output variables for each output
                const outputs: string[] = [];
                for (let i = 0; i < node.outShape.length; i++) {
                    let branchVar;
                    do {
                        branchVar = `var_${varCounter.value++}`;
                    } while (usedVarNames.has(branchVar));
                    usedVarNames.add(branchVar);
                    outputs.push(branchVar);
                }
                
                // Generate the code
                code += `${node.to_torch_functional(inputs, outputs)}\n`;
                
                // Add next nodes to stack in reverse order to process them in the original order
                for (let i = node._nexts.length - 1; i >= 0; i--) {
                    const nextNode = node._nexts[i];
                    if (nextNode) {
                        // If next node is a multi-input node, add to waiting map
                        if (nextNode instanceof MergeOp) {
                            const collectedInputs = multiInputNodes.get(nextNode) || new Array(nextNode._prevs.length).fill('');
                            const inputIndex = nextNode._prevs.findIndex(prev => prev && prev.id === node.id);
                            if (inputIndex !== -1) {
                                collectedInputs[inputIndex] = outputs[i];
                                multiInputNodes.set(nextNode, collectedInputs);
                                
                                // If we have all inputs, add to stack
                                if (collectedInputs.every(input => input !== '')) {
                                    stack.push({ node: nextNode, inputs: collectedInputs });
                                    multiInputNodes.delete(nextNode);
                                }
                            }
                        } else {
                            stack.push({ node: nextNode, inputs: [outputs[i]] });
                        }
                    }
                }
                
            } else if (node instanceof MergeOp) {
                // For MergeOp, we need to collect all inputs before processing
                const collectedInputs = multiInputNodes.get(node) || new Array(node._prevs.length).fill('');
                const inputIndex = node._prevs.findIndex(prev => prev && inputs.length > 0 && prev.id === node.id);
                
                if (inputIndex !== -1) {
                    collectedInputs[inputIndex] = inputs[0];
                    multiInputNodes.set(node, collectedInputs);
                    
                    // If we have all inputs, process the node
                    if (collectedInputs.every(input => input !== '')) {
                        code += `${outputVar} = ${node.to_torch_functional(collectedInputs)}\n`;
                        multiInputNodes.delete(node);
                        
                        // Add next node to stack if it exists
                        if (node.next) {
                            stack.push({ node: node.next, inputs: [outputVar] });
                        }
                    }
                }
            } else {
                // Regular nodes with single input/output (Op or Tensor)
                code += `${outputVar} = ${node.to_torch_functional(inputs)}\n`;
                
                // Add next node to stack if it exists
                if (node.next) {
                    // If next node is a multi-input node, add to waiting map
                    if (node.next instanceof MergeOp) {
                        const collectedInputs = multiInputNodes.get(node.next) || new Array(node.next._prevs.length).fill('');
                        const inputIndex = node.next._prevs.findIndex(prev => prev && prev.id === node.id);
                        if (inputIndex !== -1) {
                            collectedInputs[inputIndex] = outputVar;
                            multiInputNodes.set(node.next, collectedInputs);
                            
                            // If we have all inputs, add to stack
                            if (collectedInputs.every(input => input !== '')) {
                                stack.push({ node: node.next, inputs: collectedInputs });
                                multiInputNodes.delete(node.next);
                            }
                        }
                    } else {
                        stack.push({ node: node.next, inputs: [outputVar] });
                    }
                }
            }
        }
        
        // Add a return statement for the sink nodes
        const sinkNodes = Array.from(this._sinks);
        if (sinkNodes.length > 0) {
            const sinkVars = sinkNodes
                .filter(sink => !(sink instanceof MergeOp) || multiInputNodes.has(sink))
                .map(sink => {
                    if (sink instanceof MergeOp && multiInputNodes.has(sink)) {
                        const collectedInputs = multiInputNodes.get(sink)!;
                        const outputVar = `var_${varCounter.value++}`;
                        code += `${outputVar} = ${sink.to_torch_functional(collectedInputs)}\n`;
                        return outputVar;
                    }
                    return `var_${varCounter.value - 1}`; // Use the last generated variable
                });
            
            if (sinkVars.length === 1) {
                code += `return ${sinkVars[0]}\n`;
            } else if (sinkVars.length > 1) {
                code += `return (${sinkVars.join(', ')})\n`;
            }
        }
        
        return code;
    }
}







