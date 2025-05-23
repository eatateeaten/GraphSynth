import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp, Split, Copy } from './branch_op';
import { MergeOp, PointwiseOp, DotOp, CrossOp } from './merge_op';
import { Concat, PointwiseReduce } from './reduce_op';
import { assert } from './utils';
export { Tensor, Op, Concat, Split, BranchOp, MergeOp, Copy, PointwiseReduce, PointwiseOp, DotOp, CrossOp };

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

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _sources: Set<GraphNode>;
    private _sinks: Set<GraphNode>;
    private _edges: Edge[]; // Track all connections for easier disconnection

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
        this._edges = [];
    }

    /** Add a node to the graph */
    addNode(id: string, nodeType: string, params: Record<string, any>): void {
        let node: GraphNode;
        if(nodeType === "Tensor"){
            assert(params.shape, "Shape is required for Tensor");
            assert(params.variableName, "Variable name is required for Tensor");
            node = new Tensor(id, params.shape, params.variableName);
        }else if(nodeType === "Op"){
            assert(params.opType, "No operation type provided");
            node = new Op(id, params.opType, params);
        }else if(nodeType === "Split"){
            assert(params.dim !== undefined, "Dimension is required for Split");
            assert(params.sections, "Sections is required for Split");
            node = new Split(id, params.dim, params.sections, params);
        }else if(nodeType === "Concat"){
            assert(params.dim !== undefined, "Dimension is required for Concat");
            assert(params.numberOfMerges && params.numberOfMerges >= 2, "NumberOfMerges must be at least 2 for Concat");
            node = new Concat(id, params.dim, params.numberOfMerges, params);
        }else if(nodeType === "Copy"){
            assert(params.copies, "Copies parameter is required for Copy");
            node = new Copy(id, params.copies, params);
        }else if(nodeType === "PointwiseReduce"){
            assert(params.opType, "Operation type is required for PointwiseReduce");
            assert(params.numberOfMerges && params.numberOfMerges >= 2, "NumberOfMerges must be at least 2 for PointwiseReduce");
            node = new PointwiseReduce(id, params.opType, params.numberOfMerges, params);
        }else if(nodeType === "PointwiseOp"){
            assert(params.opType, "Operation type is required for PointwiseOp");
            node = new PointwiseOp(id, params.opType, params);
        }else if(nodeType === "DotOp"){
            node = new DotOp(id, params);
        }else if(nodeType === "CrossOp"){
            node = new CrossOp(id, params);
        }else{
            throw new Error(`Unknown GraphNode type: ${nodeType}`);
        }

        this._nodes.set(id, node);
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
        
        // First disconnect all connections to/from this node
        // Find all edges that involve this node
        const edgesToRemove = this._edges.filter(
            edge => edge.sourceId === nodeId || edge.sinkId === nodeId
        );
        
        // Disconnect each edge
        for (const edge of edgesToRemove) {
            try {
                // If this node is source, disconnect it from its sink
                if (edge.sourceId === nodeId && edge.sinkId) {
                    const sink = this._nodes.get(edge.sinkId);
                    if (sink) {
                        // Note: we need to handle undefined sourceIndex/sinkIndex
                        const sourceIndex = edge.sourceIndex ?? 0;
                        const sinkIndex = edge.sinkIndex ?? 0;
                        
                        // Clear the node references
                        sink.deletePrev(sinkIndex);
                        node.deleteNext(sourceIndex);
                    }
                }
                
                // If this node is sink, disconnect it from its source
                if (edge.sinkId === nodeId && edge.sourceId) {
                    const source = this._nodes.get(edge.sourceId);
                    if (source) {
                        // Note: we need to handle undefined sourceIndex/sinkIndex
                        const sourceIndex = edge.sourceIndex ?? 0;
                        const sinkIndex = edge.sinkIndex ?? 0;
                        
                        // Clear the node references
                        node.deletePrev(sinkIndex);
                        source.deleteNext(sourceIndex);
                    }
                }
            } catch (error) {
                console.warn(`Error disconnecting edge: ${error}`);
            }
        }
        
        // Remove the processed edges from the edges array
        this._edges = this._edges.filter(
            edge => edge.sourceId !== nodeId && edge.sinkId !== nodeId
        );
        
        // Remove from collections
        this._nodes.delete(nodeId);
        this._sources.delete(node);
        this._sinks.delete(node);
    }

    /**
     * Validates the source node's connection point and returns its output shape
     */
    private _validateSourceAndGetOutShape(source: GraphNode, sourceIndex: number): number[] {
        // SourceIndex must be in bounds of Source.outShape
        sourceIndex = GraphNode.checkIndexInBound(sourceIndex, source.outShapes.length, "connect");
        // Ensure the source has an output shape
        if (!source.outShapes[sourceIndex]) {
            throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
        }
        return source.outShapes[sourceIndex]!;
    }
   
    /**
     * Validates the sink node's connection point and returns its input shape, returned value can be null 
     */
    private _validateSinkAndGetInShape(sink: GraphNode, sinkIndex: number): number[] | null {
        // SourceIndex must be in bounds of Source.outShape
        sinkIndex = GraphNode.checkIndexInBound(sinkIndex, sink.inShapes.length, "connect");
        const inShape = sink.inShapes[sinkIndex];

        if(inShape === null && !GraphNode.inShapeInferred(sink))
            throw new Error(`Node with id ${sink.id} must have defined input shape`);

        return inShape;
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
    connect(sourceId: string, sinkId: string, sourceIndex: number, sinkIndex: number): void {
        // Get source node from main graph
        const source = this._nodes.get(sourceId);
        if (!source) throw new Error(`Source node with id ${sourceId} does not exist in graph`);
        // Get sink node from either main graph or pending nodes
        let sink = this._nodes.get(sinkId);
        if (!sink) throw new Error(`Sink node with id ${sinkId} does not exist in graph`);

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
        
        // Recompute sources and sinks to ensure they're correctly identified
        // This fixes potential issues with BranchOp and other node types
        this._refreshAllNodesSourceSinkStatus();
        
        // Re-check for sinks after refreshing
        if (this._sinks.size === 0) {
            throw new Error("Graph has no sink nodes after refresh");
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
            for (const nextNode of node.nexts) {
                if (nextNode) {
                    queue.push(nextNode);
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

        // Log the graph structure as an adjacency list
        this._logGraphStructure();
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
        for (const nextNode of node.nexts) {
            if (nextNode) {
                this._dfsCheckCycle(nextNode, visiting, visited);
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
    public emit_torch_functional(): string {
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
        
        // Keep track of the final variable assigned to each node
        const nodeToVarMap = new Map<string, string>();
      
        // 4) A LIFO stack for DFS. Each entry has: { node, inputs: string[] }
        //    "inputs" = the variable names feeding into this node.
        const stack: Array<{ node: GraphNode; inputs: string[] }> = [];
        
        // Helper function to update the variable mapping for a node
        const updateNodeVar = (nodeId: string, varName: string) => {
            nodeToVarMap.set(nodeId, varName);
        };
      
        // 5) A map for multi-input nodes that collects partial inputs until all are available.
        //    Key: a multi-input node, Value: array of input var names
        const waiting = new Map<GraphNode, string[]>();
      
        // 6) Assign variable names for source Tensors, push them onto the stack
        for (const source of this._sources) {
            // We already enforce that sources are Tensors in validate_graph()
            const tensorSource = source as Tensor;
          
            // Use a user-defined variableName if present, else generate one
            //let varName = tensorSource.variableName ?? newVar();
            const varName = newVar();

            usedNames.add(varName);
        
            // Track the source variable in our map
            updateNodeVar(source.id, varName);
          
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
        
                // emit_torch_functional() typically returns a snippet like: `torch.relu(VarX)` 
                // We'll do: `outVar = that_snippet`
                code += `${node.emit_torch_functional(inputs, [outVar])}\n`;
            
                // Track the output variable for this node
                updateNodeVar(node.id, outVar);
        
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
            // =============== Op (multi in, single out) ===============
            // e.g. Add, Concat, most ops with 2+ inputs but single output
                const outVar = newVar();
                usedNames.add(outVar);
        
                // Generate code as described above
                code += `${node.emit_torch_functional(inputs, [outVar])}\n`;
            
                // Track the output variable for this node
                updateNodeVar(node.id, outVar);
        
                // Pass the new single output to node's children
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
            
                // DEBUG: Log the _nexts array structure
                console.log(`DEBUG: BranchOp ${branchOp.id} _nexts array:`, 
                    branchOp.nexts.map((n, i) => n ? `[${i}]: ${n.constructor.name}[${n.id}]` : `[${i}]: null`));
            
                let numOutputs: number;
                try {
                    if (!branchOp.outShapes[0]) {
                        console.error(`ERROR: BranchOp ${branchOp.id} has no outShape defined!`);
                        throw new Error(`BranchOp ${branchOp.id} has no output shape defined`);
                    }
                
                    numOutputs = branchOp.outShapes.length;
                } catch (error: any) {
                    console.error(`ERROR in output shape for ${branchOp.id}:`, error);
                    throw new Error(`Failed to get output shape for BranchOp ${branchOp.id}: ${error.message}`);
                }
        
                // Get output variables
                const outVars: string[] = [];
                for (let i = 0; i < numOutputs; i++) {
                    const v = newVar();
                    usedNames.add(v);
                    outVars.push(v);
                }

                // Generate the torch functional code
                const branchCode = branchOp.emit_torch_functional(inputs, outVars);
                code += `${branchCode}\n`;
            
                // Track each output variable for this branch node
                for (let i = 0; i < outVars.length; i++) {
                    // For branch nodes, track variables by output index
                    // Use a unique key format: nodeId_outputIndex
                    updateNodeVar(`${branchOp.id}_${i}`, outVars[i]);
                }
            
                // IMPORTANT: Don't use filtered nextNodes - use the actual _nexts array with its indices
                // This ensures we match the correct output variable to each branch output
                for (let i = 0; i < branchOp.nexts.length; i++) {
                    const nxt = branchOp.nexts[i];
                    if (!nxt) {
                        continue;
                    }
              
                    // Make sure we don't go out of bounds in outVars array
                    if (i >= outVars.length) {
                        console.error(`ERROR: Index ${i} is out of bounds for outVars array of length ${outVars.length}`);
                        continue;
                    }
              
                    const varToPass = outVars[i];
              
                    const childInDegree = this._getPrevNodes(nxt).filter(p => p !== null).length;
                    const childSingleIn = GraphNode.singleInput(nxt);
              
                    if (childSingleIn && childInDegree <= 1) {
                        stack.push({ node: nxt, inputs: [varToPass] });
                    } else {
                        let partialInputs = waiting.get(nxt);
                        if (!partialInputs) {
                            partialInputs = new Array(childInDegree).fill("");
                        }
                        const idx = this._getPrevNodeIndex(nxt, branchOp);
                        partialInputs[idx] = varToPass;
                        waiting.set(nxt, partialInputs);
                
                        if (partialInputs.every(v => v !== "")) {
                            stack.push({ node: nxt, inputs: partialInputs });
                            waiting.delete(nxt);
                        }
                    }
                }
        
            } else {
                throw new Error(
                    `Unsupported node type (likely multi-in & multi-out) for node ID ${node.id} (${node.constructor.name}).`
                );
            }
        }

        // return statement for sink Tensors
        const sinkVars: string[] = [];
        for (const sink of this._sinks) {

            // Check if we have direct mapping for this sink
            if (nodeToVarMap.has(sink.id)) {
                sinkVars.push(nodeToVarMap.get(sink.id)!);
            } else {
                // Find nodes that connect to this sink
                const incomingNodes = [];
                for (const [nodeId, node] of this._nodes.entries()) {
                    if (node instanceof BranchOp) {
                        // Check each branch output
                        for (let i = 0; i < node.nexts.length; i++) {
                            if (node.nexts[i] === sink) {
                                incomingNodes.push({ nodeId, outputIndex: i });
                            }
                        }
                    } else if ((node instanceof Op || node instanceof MergeOp) && node.nexts[0] === sink) {
                        incomingNodes.push({ nodeId, outputIndex: 0 });
                    }
                }
            
                // Now look for mapped variables for these connections
                let foundVar = false;
                for (const { nodeId, outputIndex } of incomingNodes) {
                    const branchKey = `${nodeId}_${outputIndex}`;
                    if (nodeToVarMap.has(branchKey)) {
                        const varName = nodeToVarMap.get(branchKey)!;
                        sinkVars.push(varName);
                        foundVar = true;
                        break;
                    } else if (nodeToVarMap.has(nodeId)) {
                        const varName = nodeToVarMap.get(nodeId)!;
                        sinkVars.push(varName);
                        foundVar = true;
                        break;
                    }
                }
            
                // Fallback if no mappings found
                if (!foundVar) {
                    // Create clean names for output tensors
                    const tensorName = sink instanceof Tensor && sink.variableName ? 
                        sink.variableName : 
                        `output${sinkVars.length + 1}`;
                    sinkVars.push(tensorName);
                }
            }
        }
      
        // Return statement with all sink variables 
        if (sinkVars.length === 1) {
            code += `return ${sinkVars[0]}`;
        } else if (sinkVars.length > 1) {
            code += `return ${sinkVars.join(", ")}`;
        } else {
            code += `return None`;
        }
      
        // 10) Return the entire generated code
        return code;
    }
      
    // Helper to safely get next nodes regardless of node type
    private _getNextNodes(node: GraphNode): GraphNode[] {
        return node.nexts.filter(n => n !== null);
    }
      
    // Helper to safely get previous nodes regardless of node type
    private _getPrevNodes(node: GraphNode): GraphNode[] {
        return node.prevs.filter(p => p !== null);
    }
      
    // Helper to find the index of a prev node
    private _getPrevNodeIndex(node: GraphNode, prevNode: GraphNode): number {
        return node.prevs.findIndex(p => p && p.id === prevNode.id);
    }

    /**
     * Generates a UUID v4 string
     * @returns A UUID v4 string
     */
    public _generateUUID(): string {
        // RFC4122 compliant UUID v4
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    private _refreshAllNodesSourceSinkStatus(): void {
        // Clear existing collections
        this._sources.clear();
        this._sinks.clear();
        
        // Recompute status for all nodes
        for (const node of this._nodes.values()) {
            const isSource = !GraphNode.hasInputs(node);
            const isSink = !GraphNode.hasOutputs(node);

            if (isSource) {
                this._sources.add(node);
            }
            if (isSink) {
                this._sinks.add(node);
            }
        }
    }

    private _logGraphStructure(): void {
        console.log("========== GRAPH STRUCTURE ==========");
        
        // Create adjacency list representation
        const adjList: Record<string, {
            type: string,
            id: string,
            outEdges: {id: string, type: string, index?: number}[]
        }> = {};
        
        // Build the adjacency list
        for (const [id, node] of this._nodes.entries()) {
            const nodeType = node.constructor.name;
            adjList[id] = {
                type: nodeType,
                id: id,
                outEdges: []
            };

            node.nexts.forEach((nextNode, index) => {
                if (nextNode) {
                    adjList[id].outEdges.push({
                        id: nextNode.id,
                        type: nextNode.constructor.name,
                        index: index
                    });
                }
            });
        }
    }
}







