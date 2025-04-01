import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp } from './branch_op';
import { MergeOp} from './merge_op';
import { Concat} from './reduce_op';
import { Split, Copy } from './branch_op';

export { Tensor, Op, Concat, Split, BranchOp, MergeOp, Copy };

/**
 * Interface defining a connection edge between two nodes in the graph
 */
interface Edge {
    edgeId: string;     // Unique identifier for the edge
    sourceId: string;
    sinkId: string;
    sourceIndex?: number;
    sinkIndex?: number;
}

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
                if (!params.splitParams || params.splitParams.dim === undefined || !params.splitParams.sections) {
                    throw new Error("splitParams with dim and sections is required for Split");
                }
                node = new Split(id, params.inShape, target, params.splitParams);
                break;
                
            case "Concat":
                if (!params.concatParams || params.concatParams.dim === undefined) {
                    throw new Error("concatParams with dim is required for Concat");
                }
                /* sophia: it's probably not always 2. find a way */
                node = new Concat(id, target, params.concatParams, 2);
                break;
                
            case "Copy":
                if (!params.inShape) {
                    throw new Error("inShape parameter is required for Copy");
                }
                if (!params.copyParams || params.copyParams.copies === undefined) {
                    throw new Error("copyParams with copies is required for Copy");
                }
                node = new Copy(id, params.inShape, target, params.copyParams);
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
    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void { this._wrappedNode.addPrev(prev, prevOutShape, indexSelf, indexPrev); }
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
    private _edges: Edge[]; // Track all connections for easier disconnection

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
        this._pendingNodes = new Map();
        this._edges = [];
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
     * Validates the source node's connection point and returns its output shape
     */
    private _validateSourceAndGetOutShape(source: GraphNode, sourceIndex?: number): number[] {
        // First, ensure the source has an output shape
        if (!source.outShape) {
            throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
        }
        //------------------------------------------------------------------------------------------------
        // For nodes with multiple static outputs (Module) - WITH TENTACLES EACH WITH AN OUTPUT SHAPE
        if (GraphNode.multipleStaticOutputs(source)) {
            // SourceIndex must be defined
            if (sourceIndex === undefined) {
                throw new Error("When connecting from a node with multiple static outputs, an output index must be specified");
            }
            // SourceIndex must be in bounds of Source.outShape
            sourceIndex = GraphNode.checkIndexInBound(sourceIndex, source.outShape.length, "connect");
            // Get specific output shape
            const outShapeArray = source.outShape as number[][];
            const outputShape = outShapeArray[sourceIndex];
            // The indexed shape must be defined
            if (outputShape === null || outputShape === undefined) {
                throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id} at output ${sourceIndex}: output shape is undefined`);
            }
            return outputShape;
        } 
        //--------------------------------------------------------------------------------------------------
        // For nodes with single output (Op, Tensor, Merge)
        if (GraphNode.singleOutput(source)) {
            // Source.outShape must be defined (already checked at the top)
            return source.outShape as number[];
        }
        // Unknown node type
        throw new Error(`Unknown source node type: ${source.constructor.name}`);
    }

   
    /**
     * Validates the sink node's connection point and returns its input shape, returned value can be Null 
     */
    private _validateSinkAndGetInShape(sink: GraphNode, sinkIndex?: number): number[] | null {
        // For nodes with multiple static inputs (Module)
        if (GraphNode.multipleStaticInputs(sink)) {
            if (sinkIndex === undefined) {
                throw new Error("When connecting to a node with multiple static inputs, an input index must be specified");
            }
            if (!sink.inShape) {
                throw new Error(`Node with id ${sink.id} must have defined input shape`);
            }
            
            sinkIndex = GraphNode.checkIndexInBound(sinkIndex, sink.inShape.length, "connect");
        
            // Get specific input shape
            const inShapeArray = sink.inShape as number[][];
            if (!inShapeArray[sinkIndex]) {
                throw new Error(`Node with id ${sink.id} must have defined input shape at index ${sinkIndex}`);
            }
            return inShapeArray[sinkIndex];
        }
        
        // For nodes with multiple inputs but not static (Concat, Product, etc.)
        if (GraphNode.multipleInputs(sink) && GraphNode.inShapeInferred(sink)) {
            // These nodes handle shape validation themselves
            if (sinkIndex === undefined) {
                throw new Error("When connecting to a node with multiple inputs, an input index must be specified");
            }
            if (!sink.inShape) {
                throw new Error(`Node with id ${sink.id} must have defined input shape`);
            }
            sinkIndex = GraphNode.checkIndexInBound(sinkIndex, sink.inShape.length, "connect");
            return null //return null shape it can be cascaded and checked for shape consistancy for other ports 
        }

        //---------------
        // For nodes with single input and not inShapeInferred (Tensor)
        if (GraphNode.singleInput(sink) && !GraphNode.inShapeInferred(sink)) {
            if (!sink.inShape) {
                throw new Error(`${sink.constructor.name} with id ${sink.id} must have defined input shape`);
            }
            return sink.inShape as number[];
        }
        
        // For nodes with single input and inShapeInferred (Op)
        if (GraphNode.singleInput(sink) && GraphNode.inShapeInferred(sink)) {
            // These nodes infer their shape from the connection
            return null; // Skip shape check
        }
        
        // Unknown node type
        throw new Error(`Unknown sink node type: ${sink.constructor.name}`);
    }

    /**
     * Connects two nodes.
     * 
     * This method establishes a connection between two nodes in the graph.
     * If the source node exists in the main graph and the sink node is pending,
     * the sink node will be moved from pending to the main graph.
     * Shape validation is performed to ensure compatible connections.
     * 
     * @param sourceId - ID of the source node
     * @param sinkId - ID of the sink node
     * @param sourceIndex - Index for the source node output (required for nodes with multiple outputs)
     * @param sinkIndex - Index for the sink node input (required for nodes with multiple inputs)
     * @throws Error if nodes don't exist or have incompatible shapes
     * 
     * @example
     * // Connect two nodes
     * graph.connect("opId", "tensorId");
     * 
     * // Connect with specific indices (for branch/merge operations)
     * graph.connect("splitId", "opId", 0); // Connect from first output of split
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
        //-------------------------------------------------------
        // Validate connection endpoints and get shapes
        const sourceOutShape = this._validateSourceAndGetOutShape(source, sourceIndex);
        const sinkInShape = this._validateSinkAndGetInShape(sink, sinkIndex);

        // Skip shape compatibility check if sink has no defined input shape yet
        // This allows connecting nodes before their shapes are fully determined
        if (sinkInShape !== null) {
            if (!GraphNode.shapeMatch(sourceOutShape, sinkInShape)) {
                throw new Error(`Shape mismatch: Cannot connect ${source.constructor.name} with output shape [${sourceOutShape}] to ${sink.constructor.name} with input shape [${sinkInShape}]`);
            }
        }
        //---------------------------------------------------------
        // Establish bidirectional connections
        // Let each node handle its own connection logic
        sink.addPrev(source, sourceOutShape, sinkIndex, sourceIndex);  ///this is where inShapeInferred Nodes must check for validity of inferring inshape from prev outShape 
        source.addNext(sink, sourceIndex, sinkIndex);

        // Add the connection to our edge list with a unique ID
        this._edges.push({
            edgeId: this._generateUUID(),
            sourceId: sourceId,
            sinkId: sinkId,
            sourceIndex: sourceIndex,
            sinkIndex: sinkIndex
        });
        
        //---------------------------------------------------------
        // Only if the connection was successful and sink was pending, move it to the main graph
        if (sinkIsPending) {
            this._pendingNodes.delete(sink.id);
            this._nodes.set(sink.id, sink);
        }
        //----------------------------------------------------------
        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    
    disconnect(sourceId: string, sinkId: string, sourceIndex: number, sinkIndex: number): void {        
        const source = this._nodes.get(sourceId);
        const sink = this._nodes.get(sinkId);

        if(!source) throw new Error(`No source with ID ${sourceId}`);
        if(!sink) throw new Error(`No sink with ID ${sinkId}`);

        sink.deletePrev(sinkIndex);
        source.deleteNext(sourceIndex);

        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }


    private _refreshNodeSinkSourceStatus(node: GraphNode): void {
        // Check source status (no incoming connections)
        const isSource = !GraphNode.hasInputs(node);
            
        // Check sink status (no outgoing connections)
        const isSink = !GraphNode.hasOutputs(node);

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
    public to_torch_functional(): string {
        // 1) Validate the graph (checks for sources, sinks, shape consistency, cycles, etc.)
        this.validate_graph();
      
        // 2) Prepare a buffer for the generated code
        let code = "";
      
        // 3) A variable counter to produce unique variable names
        let varCounter = 0;
        function newVar(): string {
          return `Var${varCounter++}`;
        }
      
        // Track which variable names we've used (just in case you need to avoid collision with user-provided ones)
        const usedNames = new Set<string>();
      
        // 4) A LIFO stack for DFS. Each entry has: { node, inputs: string[] }
        //    "inputs" = the variable names feeding into this node.
        const stack: Array<{ node: GraphNode; inputs: string[] }> = [];
      
        // 5) A map for multi-input nodes that collects partial inputs until all are available.
        //    Key: a multi-input node, Value: array of input var names
        const waiting = new Map<GraphNode, string[]>();
      
        // 6) Assign variable names for source Tensors, push them onto the stack
        for (const source of this._sources) {
          // We already enforce that sources are Tensors in validate_graph()
          const tensorSource = source as Tensor;
          
          // Use a user-defined variableName if present, else generate one
          let varName = tensorSource.variableName ?? newVar();
          while (usedNames.has(varName)) {
            varName = newVar();
          }
          usedNames.add(varName);
        
          // Generate code line for creating/referencing this Tensor source
          // Typically something like: `Var0 = torch.randn([...])` or `Var0 = input_image`
          code += `${varName} = ${tensorSource.to_torch_functional([], [varName])}\n`;
        
          // Send this var to the source's children
          const nextNodes = this._getNextNodes(source);
          for (const nxt of nextNodes) {
            if (!nxt) continue;
            
            // Count how many inputs the child has
            const inDegree = this._getPrevNodes(nxt).filter(p => p !== null).length;
            const singleIn = GraphNode.singleInput(nxt);
        
            if (singleIn && inDegree <= 1) {
              // Single-input node, safe to push directly
              stack.push({ node: nxt, inputs: [varName] });
            } else {
              // Multi-input node → partially fill in waiting
              let partialInputs = waiting.get(nxt);
              if (!partialInputs) {
                partialInputs = new Array(inDegree).fill("");
              }
              // Find the correct index for this edge
              const idx = this._getPrevNodeIndex(nxt, source);
              partialInputs[idx] = varName;
              waiting.set(nxt, partialInputs);
        
              // If all inputs are now filled, push the node onto stack
              if (partialInputs.every(v => v !== "")) {
                stack.push({ node: nxt, inputs: partialInputs });
                waiting.delete(nxt);
              }
            }
          }
        }
      
        // 7) Keep track of visited nodes so we don't generate code twice
        const visited = new Set<GraphNode>();
      
        // 8) DFS: pop from the stack, generate code, push outputs forward
        while (stack.length > 0) {
          const { node, inputs } = stack.pop()!;
        
          if (visited.has(node)) {
            // Already handled this node
            continue;
          }
          visited.add(node);
        
          // Determine node "type" by singleInput/singleOutput
          const singleIn = GraphNode.singleInput(node);
          const singleOut = GraphNode.singleOutput(node);
        
          if (singleIn && singleOut) {
            // =============== Op or Tensor (single in, single out) ===============
            // (e.g. ReLU, any standard unary/binary op, or an in-graph Tensor)
            const outVar = newVar();
            usedNames.add(outVar);
        
            // to_torch_functional() typically returns a snippet like: `torch.relu(VarX)` 
            // We'll do: `outVar = that_snippet`
            code += `${outVar} = ${node.to_torch_functional(inputs, [outVar])}\n`;
        
            // Pass outVar to this node's children
            const nextNodes = this._getNextNodes(node);
            for (const nxt of nextNodes) {
              if (!nxt) continue;
              const childInDegree = this._getPrevNodes(nxt).filter(p => p !== null).length;
              const childSingleIn = GraphNode.singleInput(nxt);
              
              if (childSingleIn && childInDegree <= 1) {
                stack.push({ node: nxt, inputs: [outVar] });
              } else {
                // Multi-input child
                let partialInputs = waiting.get(nxt);
                if (!partialInputs) {
                  partialInputs = new Array(childInDegree).fill("");
                }
                const idx = this._getPrevNodeIndex(nxt, node);
                partialInputs[idx] = outVar;
                waiting.set(nxt, partialInputs);
        
                if (partialInputs.every(v => v !== "")) {
                  stack.push({ node: nxt, inputs: partialInputs });
                  waiting.delete(nxt);
                }
              }
            }
        
          } else if (!singleIn && singleOut) {
            // =============== Merge (multi-input, single output) ===============
            // e.g. Concat, AddN, etc.
            // `inputs` array should have all inDegree vars
            const outVar = newVar();
            usedNames.add(outVar);
        
            code += `${outVar} = ${node.to_torch_functional(inputs, [outVar])}\n`;
        
            // Pass output var to children
            const nextNodes = this._getNextNodes(node);
            for (const nxt of nextNodes) {
              if (!nxt) continue;
              const childInDegree = this._getPrevNodes(nxt).filter(p => p !== null).length;
              const childSingleIn = GraphNode.singleInput(nxt);
        
              if (childSingleIn && childInDegree <= 1) {
                stack.push({ node: nxt, inputs: [outVar] });
              } else {
                let partialInputs = waiting.get(nxt);
                if (!partialInputs) {
                  partialInputs = new Array(childInDegree).fill("");
                }
                const idx = this._getPrevNodeIndex(nxt, node);
                partialInputs[idx] = outVar;
                waiting.set(nxt, partialInputs);
        
                if (partialInputs.every(v => v !== "")) {
                  stack.push({ node: nxt, inputs: partialInputs });
                  waiting.delete(nxt);
                }
              }
            }
        
          } else if (singleIn && !singleOut) {
            // =============== Branch (single input, multiple outputs) ===============
            // e.g. Split, Copy, or any node that fans out multiple paths
            // The node's `outShape` typically has the # of outputs.
            const branchOp = node as BranchOp;  // or a node that acts like Branch
            const numOutputs = branchOp.outShape.length;
        
            const outVars: string[] = [];
            for (let i = 0; i < numOutputs; i++) {
              const v = newVar();
              usedNames.add(v);
              outVars.push(v);
            }
        
            // e.g. `[Var2, Var3] = torch.split(Var1, ...)`
            code += `${node.to_torch_functional(inputs, outVars)}\n`;
        
            // Connect each outVar to the corresponding child (by index)
            const nextNodes = this._getNextNodes(node);
            for (let i = 0; i < nextNodes.length; i++) {
              const nxt = nextNodes[i];
              if (!nxt) continue;
              const childInDegree = this._getPrevNodes(nxt).filter(p => p !== null).length;
              const childSingleIn = GraphNode.singleInput(nxt);
        
              const varToPass = outVars[i];
        
              if (childSingleIn && childInDegree <= 1) {
                stack.push({ node: nxt, inputs: [varToPass] });
              } else {
                let partialInputs = waiting.get(nxt);
                if (!partialInputs) {
                  partialInputs = new Array(childInDegree).fill("");
                }
                const idx = this._getPrevNodeIndex(nxt, node);
                partialInputs[idx] = varToPass;
                waiting.set(nxt, partialInputs);
        
                if (partialInputs.every(v => v !== "")) {
                  stack.push({ node: nxt, inputs: partialInputs });
                  waiting.delete(nxt);
                }
              }
            }
        
          } else {
            // =============== Multi-input, multi-output? Or unrecognized module? ===============
            throw new Error(
              `Unsupported node type (likely multi-in & multi-out) for node ID ${node.id} (${node.constructor.name}).`
            );
          }
        }
      
        // 9) Finally, add a return statement for sink Tensors
        //    We assume all sinks are Tensors with single output. So we produce their final var name.
        //    In practice, you must store each node's final assigned variable name in the loop above.
        //    For demonstration, we do a placeholder or a minimal approach:
        const sinkVars: string[] = [];
        for (const sink of this._sinks) {
          // If we had stored the last-assigned variable in a map, we'd retrieve it here.
          // For example: `const finalVar = finalVarMap.get(sink.id)`
          // But here we'll just stub it out:
          sinkVars.push(`#finalVar_for_${sink.id}#`);
        }
      
        if (sinkVars.length === 1) {
          code += `return ${sinkVars[0]}\n`;
        } else if (sinkVars.length > 1) {
          code += `return (${sinkVars.join(', ')})\n`;
        }
      
        // 10) Return the entire generated code
        return code;
      }
      
      // Helper to safely get next nodes regardless of node type
      private _getNextNodes(node: GraphNode): GraphNode[] {
        if (node instanceof BranchOp) {
          return (node as BranchOp)._nexts.filter(n => n !== null);
        } else {
          return node.next ? [node.next] : [];
        }
      }
      
      // Helper to safely get previous nodes regardless of node type
      private _getPrevNodes(node: GraphNode): GraphNode[] {
        if (node instanceof MergeOp) {
          return (node as MergeOp)._prevs.filter(p => p !== null);
        } else {
          return node.prev ? [node.prev] : [];
        }
      }
      
      // Helper to find the index of a prev node
      private _getPrevNodeIndex(node: GraphNode, prevNode: GraphNode): number {
        if (node instanceof MergeOp) {
          return (node as MergeOp)._prevs.findIndex(p => p && p.id === prevNode.id);
        } else {
          return 0; // For single-input nodes
        }
      }

    // Add a helper method to generate UUIDs
    /**
     * Generates a UUID v4 string
     * @private
     * @returns A UUID v4 string
     */
    private _generateUUID(): string {
        // RFC4122 compliant UUID v4
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
}







