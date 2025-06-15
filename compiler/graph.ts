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
    sourcePortIndex?: number;
    sinkPortIndex?: number;
}

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _sources: Set<GraphNode>;
    private _sinks: Set<GraphNode>;
    private _edges: Edge[]; // Track all connections for easier disconnection
    
    // Factory functions for creating different node types
    private _nodeFactories: Map<string, (id: string, params: Record<string, any>) => GraphNode>;

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
        this._edges = [];
        
        // Initialize factory functions
        this._nodeFactories = new Map<string, (id: string, params: Record<string, any>) => GraphNode>([
            ['Tensor', (id, params) => {
                assert(params.shape, "Shape is required for Tensor");
                assert(params.variableName, "Variable name is required for Tensor");
                return new Tensor(id, params.shape, params.variableName);
            }],
            ['Op', (id, params) => {
                assert(params.opType, "No operation type provided");
                return new Op(id, params.opType, params);
            }],
            ['Split', (id, params) => {
                assert(params.dim !== undefined, "Dimension is required for Split");
                assert(params.sections, "Sections is required for Split");
                return new Split(id, params.dim, params.sections, params);
            }],
            ['Concat', (id, params) => {
                assert(params.dim !== undefined, "Dimension is required for Concat");
                assert(params.numberOfMerges && params.numberOfMerges >= 2, "NumberOfMerges must be at least 2 for Concat");
                return new Concat(id, params.dim, params.numberOfMerges, params);
            }],
            ['Copy', (id, params) => {
                assert(params.copies, "Copies parameter is required for Copy");
                return new Copy(id, params.copies, params);
            }],
            ['PointwiseReduce', (id, params) => {
                assert(params.opType, "Operation type is required for PointwiseReduce");
                assert(params.numberOfMerges && params.numberOfMerges >= 2, "NumberOfMerges must be at least 2 for PointwiseReduce");
                return new PointwiseReduce(id, params.opType, params.numberOfMerges, params);
            }],
            ['PointwiseOp', (id, params) => {
                assert(params.opType, "Operation type is required for PointwiseOp");
                return new PointwiseOp(id, params.opType, params);
            }],
            ['DotOp', (id, params) => {
                return new DotOp(id, params);
            }],
            ['CrossOp', (id, params) => {
                return new CrossOp(id, params);
            }]
        ]);
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


    /** Add a node to the graph */
    addNode(id: string, nodeType: string, params: Record<string, any>): void {
        const factory = this._nodeFactories.get(nodeType);
        if (!factory) {
            throw new Error(`Unknown GraphNode type: ${nodeType}`);
        }
        
        const node = factory(id, params);
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
        throw new Error("removeNode is not implemented");
    }


    /**
     * Validates the source node's connection port and returns its output shape
     */
    private _validateSourcePortAndGetShape(source: GraphNode, sourcePortIndex: number): number[] {
        // SourcePortIndex must be in bounds of Source.outShape
        sourcePortIndex = GraphNode.isIndexInBound(sourcePortIndex, source.outShapes.length, "connect");
        // The source port output shape has to be defined, whether the current node applies inShape inference or not 
        if (!source.outShapes[sourcePortIndex]) {
            throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
        } else {
            return source.outShapes[sourcePortIndex]!;
        }

    }
   
    /**
     * Validates the sink node's connection port and returns its input shape, returned value can be null if sink node has input shape inferred
     */
    private _validateSinkPortAndGetShape(sink: GraphNode, sinkPortIndex: number): number[] | null {
        // SinkPortIndex must be in bounds of Sink.inShapes
        sinkPortIndex = GraphNode.isIndexInBound(sinkPortIndex, sink.inShapes.length, "connect");
        // Check if sink port has a defined input shape or if shape inference is allowed
        if (sink.inShapes[sinkPortIndex] === null && !GraphNode.inShapeInferred(sink)) {  //sink node does not have input shape inferred 
            throw new Error(`Node with id ${sink.id} must have defined input shape`);
        } else {
            return sink.inShapes[sinkPortIndex];
        }
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
     * @param sourcePortIndex - Index for the source node output (required for nodes with multiple outputs)
     * @param sinkPortIndex - Index for the sink node input (required for nodes with multiple inputs)
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
    connect(sourceId: string, sinkId: string, sourcePortIndex: number, sinkPortIndex: number): void {
        // ------ Check both source and sink exist --------
        const source = this._nodes.get(sourceId);
        if (!source) throw new Error(`Source node with id ${sourceId} does not exist in graph`);
        const sink = this._nodes.get(sinkId);
        if (!sink) throw new Error(`Sink node with id ${sinkId} does not exist in graph`);

        // ------ Validate indices of connection ports and get source and sink port shapes
        const sourcePortOutShape = this._validateSourcePortAndGetShape(source, sourcePortIndex);
        const sinkPortInShape = this._validateSinkPortAndGetShape(sink, sinkPortIndex);
        
        // ------ Shape-match Check --------
        // Skip shape compatibility check if sink has no defined input shape yet
        if (sinkPortInShape !== null) {
            if (!GraphNode.shapeMatch(sourcePortOutShape, sinkPortInShape)) {
                throw new Error(`Shape mismatch: Cannot connect ${source.constructor.name} with output shape [${sourcePortOutShape}] to ${sink.constructor.name} with input shape [${sinkPortInShape}]`);
            }
        }

        // ------- Establish bidirectional references -------
        // let each node handle its connection setup 
        sink.addPrev(source, sourcePortOutShape, sinkPortIndex, sourcePortIndex);  
        // this is where inShapeInferred Nodes must check for validity of inferring inShape from prev outShape 
        // If inShape inferred from prev outShape is not valid for this node's operation, such as applying to Conv(3, 3) shape of (2, 2) without padding, it will promptly fail 

        source.addNext(sink, sourcePortIndex, sinkPortIndex);

        // -----  Add the connection to the edge list ------
        // with a unique ID 
        this._edges.push({
            edgeId: this._generateUUID(),
            sourceId: sourceId,
            sinkId: sinkId,
            sourcePortIndex: sourcePortIndex,
            sinkPortIndex: sinkPortIndex
        });

        //----------------------------------------------------------
        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    
    /**
     * Disconnects two nodes by removing their connection.
     * 
     * This method removes the bidirectional connection between two nodes,
     * cleans up the edge tracking, and updates the graph's source/sink status.
     * 
     * @param sourceId - ID of the source node
     * @param sinkId - ID of the sink node  
     * @param sourcePortIndex - Index of the source node output port
     * @param sinkPortIndex - Index of the sink node input port
     * @throws Error if either node doesn't exist in the graph
     * 
     * @example
     * // Disconnect two nodes
     * graph.disconnect("sourceId", "sinkId", 0, 0);
     * 
     * // Disconnect specific ports for multi-port nodes
     * graph.disconnect("splitId", "concatId", 1, 2);
     */
    disconnect(sourceId: string, sinkId: string, sourcePortIndex: number, sinkPortIndex: number): void {        
        const source = this._nodes.get(sourceId);
        const sink = this._nodes.get(sinkId);

        if (!source) throw new Error(`No source with ID ${sourceId}`);
        if (!sink) throw new Error(`No sink with ID ${sinkId}`);

        // Remove bidirectional node connections
        sink.deletePrev(sinkPortIndex); //TODO check each class implemetation for correctness 
        source.deleteNext(sourcePortIndex); //TODO check each class implemetation for correctness 

        // Remove the corresponding edge from the edge list 
        this._edges = this._edges.filter(edge => 
            !(edge.sourceId === sourceId && edge.sinkId === sinkId && 
              edge.sourcePortIndex === sourcePortIndex && edge.sinkPortIndex === sinkPortIndex)
        );

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

//--------- ALREADY WENT THROUGH EVERYTHING ABOVE --------- 
    validate_graph(): void {
        //check the graph has source and sink tensors 
        //check that all sources and sinks are tensors 
        //check that the entire graph is connected 
        //

        // Recompute sources and sinks to ensure they're correctly identified
        // This fixes potential issues with BranchOp and other node types
        this._refreshAllNodesSourceSinkStatus();
        
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
     * Generates complete SSA (Static Single Assignment) style PyTorch code from the graph DAG.
     * Each variable is assigned exactly once with automatic variable name generation.
     * Uses a depth-first traversal strategy to respect computation order.
     * 
     * @returns A string containing complete PyTorch code in SSA functional style
     */
    public emitTorchFunctional(): string {
        // 1) Validate the graph (checks for sources, sinks, shape consistency, cycles, etc.)
        this.validate_graph();
      
        // 2) SSA variable management
        let varCounter = 0;
        const newVar = (): string => `v${varCounter++}`;
        
        // Map each node to its output variable(s) - for branch nodes, use nodeId_outputIndex
        const nodeOutputVars = new Map<string, string>();
        
        // 3) Topological sort to get proper execution order
        const topoOrder = this._topologicalSort();
        
        // 4) Generate SSA code
        let code = "";
        
        // Process each node in topological order
        for (const node of topoOrder) {
            if (this._sources.has(node)) {
                // Source nodes (Tensors) - assign initial variables
                const sourceVar = newVar();
                nodeOutputVars.set(node.id, sourceVar);
                
                // For source tensors, we might want to use their variableName if available
                const tensorSource = node as Tensor;
                const inputName = tensorSource.variableName || `input_${node.id}`;
                code += `${sourceVar} = ${inputName}  # Source: ${node.id}\n`;
                
            } else if (GraphNode.singleInput(node) && GraphNode.singleOutput(node)) {
                // Single input, single output (Op, most operations)
                const inputVar = this._getSSAInputVariable(node, nodeOutputVars);
                const outputVar = newVar();
                nodeOutputVars.set(node.id, outputVar);
                
                // Get the operation code from the node (without variable names)
                const opCode = this._getSSAOperationCode(node);
                code += `${outputVar} = ${opCode}(${inputVar})  # ${node.constructor.name}: ${node.id}\n`;
                
            } else if (!GraphNode.singleInput(node) && GraphNode.singleOutput(node)) {
                // Multiple inputs, single output (MergeOp, Add, Concat, etc.)
                const inputVars = this._getSSAInputVariables(node, nodeOutputVars);
                const outputVar = newVar();
                nodeOutputVars.set(node.id, outputVar);
                
                // Get the operation code from the node
                const opCode = this._getSSAOperationCode(node);
                code += `${outputVar} = ${opCode}(${inputVars.join(', ')})  # ${node.constructor.name}: ${node.id}\n`;
                
            } else if (GraphNode.singleInput(node) && !GraphNode.singleOutput(node)) {
                // Single input, multiple outputs (BranchOp, Split, Copy)
                const inputVar = this._getSSAInputVariable(node, nodeOutputVars);
                const branchOp = node as BranchOp;
                const numOutputs = branchOp.outShapes.length;
                
                const outputVars: string[] = [];
                for (let i = 0; i < numOutputs; i++) {
                    const outVar = newVar();
                    outputVars.push(outVar);
                    // Store each output with indexed key for branch operations
                    nodeOutputVars.set(`${node.id}_${i}`, outVar);
                }
                
                // Get the operation code from the node
                const opCode = this._getSSABranchOperationCode(node);
                if (outputVars.length === 1) {
                    code += `${outputVars[0]} = ${opCode}(${inputVar})  # ${node.constructor.name}: ${node.id}\n`;
                } else {
                    code += `${outputVars.join(', ')} = ${opCode}(${inputVar})  # ${node.constructor.name}: ${node.id}\n`;
                }
                
            } else {
                throw new Error(
                    `Unsupported node type (multi-in & multi-out) for node ID ${node.id} (${node.constructor.name})`
                );
            }
        }
        
        // 5) Generate return statement for sink variables
        const sinkVars: string[] = [];
        for (const sink of this._sinks) {
            // Find the variable that feeds into this sink
            const sinkVar = this._getSSASinkVariable(sink, nodeOutputVars);
            sinkVars.push(sinkVar);
        }
        
        if (sinkVars.length > 0) {
            code += `return ${sinkVars.join(', ')}\n`;
        }
        
        return code;
    }

    // Helper methods for SSA code generation
    private _topologicalSort(): GraphNode[] {
        const visited = new Set<GraphNode>();
        const result: GraphNode[] = [];
        
        const visit = (node: GraphNode) => {
            if (visited.has(node)) return;
            visited.add(node);
            
            // Visit all dependencies first
            for (const prev of node.prevs) {
                if (prev) visit(prev);
            }
            
            result.push(node);
        };
        
        // Start from all nodes (sources will naturally be first due to no dependencies)
        for (const node of this._nodes.values()) {
            visit(node);
        }
        
        return result;
    }
    
    private _getSSAInputVariable(node: GraphNode, nodeOutputVars: Map<string, string>): string {
        const prevNodes = this._getPrevNodes(node);
        const prev = prevNodes.find(p => p !== null);
        if (!prev) {
            throw new Error(`Node ${node.id} has no input connections`);
        }
        
        // Check if it's a branch output
        if (prev instanceof BranchOp) {
            const prevNodeIndex = this._getPrevNodeIndex(node, prev);
            const key = `${prev.id}_${prevNodeIndex}`;
            if (nodeOutputVars.has(key)) {
                return nodeOutputVars.get(key)!;
            }
        }
        
        // Regular single output node
        const inputVar = nodeOutputVars.get(prev.id);
        if (!inputVar) {
            throw new Error(`No output variable found for input node ${prev.id}`);
        }
        return inputVar;
    }
    
    private _getSSAInputVariables(node: GraphNode, nodeOutputVars: Map<string, string>): string[] {
        const prevNodes = this._getPrevNodes(node);
        const inputVars: string[] = [];
        
        for (let i = 0; i < prevNodes.length; i++) {
            const prev = prevNodes[i];
            if (!prev) continue;
            
            // Check if it's a branch output
            if (prev instanceof BranchOp) {
                const key = `${prev.id}_${i}`;
                if (nodeOutputVars.has(key)) {
                    inputVars.push(nodeOutputVars.get(key)!);
                } else {
                    throw new Error(`No output variable found for branch ${prev.id} output ${i}`);
                }
            } else {
                // Regular single output node
                const inputVar = nodeOutputVars.get(prev.id);
                if (!inputVar) {
                    throw new Error(`No output variable found for input node ${prev.id}`);
                }
                inputVars.push(inputVar);
            }
        }
        
        return inputVars;
    }
    
    private _getSSAOperationCode(node: GraphNode): string {
        // Get the raw operation code without variable assignments
        if (node instanceof Op) {
            return node.emitTorch(); // Use the parameter-less version
        } else if (node instanceof MergeOp) {
            // For merge ops, we need to extract the operation from emitTorchFunctional
            // This is a temporary solution - ideally MergeOp should have emitTorch() too
            const tempCode = node.emitTorchFunctional(['temp1', 'temp2'], ['tempOut']);
            // Extract the operation part (everything after the '=')
            const match = tempCode.match(/= (.+)$/);
            return match ? match[1] : tempCode;
        }
        throw new Error(`Unsupported node type for operation code: ${node.constructor.name}`);
    }
    
    private _getSSABranchOperationCode(node: GraphNode): string {
        if (node instanceof BranchOp) {
            // For branch ops, we need to extract the operation from emitTorchFunctional
            const tempCode = node.emitTorchFunctional(['tempIn'], ['tempOut1', 'tempOut2']);
            // Extract the operation part (everything after the '=')
            const match = tempCode.match(/= (.+)$/);
            return match ? match[1] : tempCode;
        }
        throw new Error(`Unsupported node type for branch operation code: ${node.constructor.name}`);
    }
    
    private _getSSASinkVariable(sink: GraphNode, nodeOutputVars: Map<string, string>): string {
        // Find the node that connects to this sink
        for (const [nodeId, node] of this._nodes.entries()) {
            if (node instanceof BranchOp) {
                // Check each branch output
                for (let i = 0; i < node.nexts.length; i++) {
                    if (node.nexts[i] === sink) {
                        const key = `${nodeId}_${i}`;
                        if (nodeOutputVars.has(key)) {
                            return nodeOutputVars.get(key)!;
                        }
                    }
                }
            } else if ((node instanceof Op || node instanceof MergeOp) && node.nexts[0] === sink) {
                if (nodeOutputVars.has(nodeId)) {
                    return nodeOutputVars.get(nodeId)!;
                }
            }
        }
        
        // Fallback - use sink's variable name if it's a tensor
        if (sink instanceof Tensor && sink.variableName) {
            return sink.variableName;
        }
        
        throw new Error(`No input variable found for sink ${sink.id}`);
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







