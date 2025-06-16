import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp, Split, Copy } from './branch_op';
import { MergeOp, PointwiseOp, DotOp, CrossOp } from './merge_op';
import { Concat, PointwiseReduce } from './reduce_op';
import { isNodeType, NodeType, ParamError, Shape } from "./types";
export { Tensor, Op, Concat, Split, BranchOp, MergeOp, Copy, PointwiseReduce, PointwiseOp, DotOp, CrossOp };

/// Node name to node builder lookup
const moduleFromParams: Record<string, (id: NodeType, params: Record<string, any>) => GraphNode> = {
    "Tensor": Tensor.fromParams,
    "Op": Op.fromParams,
    "Split": Split.fromParams,
    "Concat": Concat.fromParams,
    "Copy": Copy.fromParams,
    "PointwiseReduce": PointwiseReduce.fromParams,
    "PointwiseOp": PointwiseOp.fromParams,
    "DotOp": DotOp.fromParams,
    "CrossOp": CrossOp.fromParams,
};

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

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
        this._edges = [];
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
        if(!isNodeType(nodeType))
            throw new Error(`Unknown GraphNode type: ${nodeType}`);
    
        const factory = moduleFromParams[nodeType];
        
        const node = factory(id as NodeType, params);
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
        for (const source of this._sources) {
            if (!(source instanceof Tensor)) {
                throw new Error(`Source node ${source.id} is not a Tensor (found ${source.constructor.name} instead)`);
            }
        }
        // Check that all sink nodes are Tensors
        for (const sink of this._sinks) {
            if (!(sink instanceof Tensor)) {
                throw new Error(`Sink node ${sink.id} is not a Tensor (found ${sink.constructor.name} instead)`);
            }
        }
        
        // Check that all sinks are reachable from sources and that the graph doesn't contain cycles using Kahn's algorithm 
        this. _bfsValidateGraph()
        // Log the graph structure as an adjacency list
        this._logGraphStructure();
    }

//--------- ALREADY WENT THROUGH EVERYTHING ABOVE --------- 

    private _bfsValidateGraph(): void {
        // Calculate in-degrees for all nodes
        const inDegree = new Map<string, number>();
        for (const [id, node] of this._nodes) {
            inDegree.set(id, node.prevs.filter(p => p !== null).length);
        }
        
        // Start with nodes that have no incoming edges (sources)
        const queue: GraphNode[] = [];
        for (const [id, degree] of inDegree) {
            if (degree === 0) {
                queue.push(this._nodes.get(id)!);
            }
        }
        
        const visited = new Set<string>();
        
        while (queue.length > 0) {
            const node = queue.shift()!;
            visited.add(node.id);
            
            // Process all outgoing edges
            for (const nextNode of node.nexts) {
                if (!nextNode) continue;
                
                // Decrease in-degree
                const newDegree = inDegree.get(nextNode.id)! - 1;
                inDegree.set(nextNode.id, newDegree);
                
                // If in-degree becomes 0, add to queue
                if (newDegree === 0) {
                    queue.push(nextNode);
                }
            }
        }
        
        // Check both reachability AND cycles in one shot!
        if (visited.size !== this._nodes.size) {
            const unprocessed = Array.from(this._nodes.keys())
                .filter(id => !visited.has(id));
            
            // If we couldn't process all nodes, there's either:
            // 1. A cycle (nodes with non-zero in-degree remaining)
            // 2. Unreachable nodes
            
            const cycleNodes = unprocessed.filter(id => inDegree.get(id)! > 0);
            if (cycleNodes.length > 0) {
                throw new Error(`Graph contains cycles involving nodes: ${cycleNodes.join(', ')}`);
            } else {
                throw new Error(`Graph contains unreachable nodes: ${unprocessed.join(', ')}`);
            }
        }
    }
    
    /**
     * Generates intermediate representation (IR) for the entire graph.
     * Shows the structure and data flow of all nodes.
     * 
     * @returns A string containing the IR representation of the graph
     */
    public emitIR(): string {
        this.validate_graph();
        
        const topoOrder = this._topologicalSort();
        let ir = "=== GRAPH IR ===\n";
        
        for (const node of topoOrder) {
            const nodeIR = node.emitIR();
            ir += `${node.id}: ${nodeIR}\n`;
        }
        
        // Add connection information
        ir += "\n=== CONNECTIONS ===\n";
        for (const edge of this._edges) {
            ir += `${edge.sourceId}[${edge.sourcePortIndex}] -> ${edge.sinkId}[${edge.sinkPortIndex}]\n`;
        }
        
        return ir;
    }
    /**
     * Generates complete SSA (Static Single Assignment) style PyTorch code from the graph DAG.
     * Each variable is assigned exactly once with automatic variable name generation.
     * Uses a depth-first traversal strategy to respect computation order.
     * 
     * @returns A string containing complete PyTorch code in SSA functional style
     */
    public emitTorchFunctional(): string {
        this.validate_graph();
        
        let varCounter = 0;
        const newVar = () => `v${varCounter++}`;
        const nodeVars = new Map<string, string>();
        const topoOrder = this._topologicalSort();
        
        let code = "";
        
        for (const node of topoOrder) {
            const nodeCode = this._generateNodeCode(node, nodeVars, newVar);
            code += nodeCode;
        }
        
        // Generate return statement
        const sinkVars = this._sinks.size > 0 
            ? Array.from(this._sinks).map(sink => this._getSinkVar(sink, nodeVars))
            : [];
            
        if (sinkVars.length > 0) {
            code += `return ${sinkVars.join(', ')}\n`;
        }
        
        return code;
    }



    private _generateNodeCode(node: GraphNode, nodeVars: Map<string, string>, newVar: () => string): string {
        if (this._sources.has(node)) {
            return this._generateSourceCode(node, nodeVars, newVar);
        }
        
        const inputs = this._getNodeInputs(node, nodeVars);
        const outputs = this._createNodeOutputs(node, nodeVars, newVar);
        const operation = this._getNodeOperation(node);
        
        return this._formatNodeCode(node, inputs, outputs, operation);
    }

    private _generateSourceCode(node: GraphNode, nodeVars: Map<string, string>, newVar: () => string): string {
        const outputVar = newVar();
        nodeVars.set(node.id, outputVar);
        
        const tensor = node as Tensor;
        const inputName = tensor.variableName || `input_${node.id}`;
        
        return `${outputVar} = ${inputName}  # Source: ${node.id}\n`;
    }

    private _getNodeInputs(node: GraphNode, nodeVars: Map<string, string>): string[] {
        const prevNodes = this._getPrevNodes(node);
        const inputs: string[] = [];
        
        for (let i = 0; i < prevNodes.length; i++) {
            const prev = prevNodes[i];
            if (!prev) continue;
            
            const inputVar = prev instanceof BranchOp 
                ? nodeVars.get(`${prev.id}_${i}`)
                : nodeVars.get(prev.id);
                
            if (!inputVar) {
                throw new Error(`No input variable found for ${prev.id}`);
            }
            inputs.push(inputVar);
        }
        
        return inputs;
    }

    private _createNodeOutputs(node: GraphNode, nodeVars: Map<string, string>, newVar: () => string): string[] {
        if (GraphNode.singleOutput(node)) {
            const outputVar = newVar();
            nodeVars.set(node.id, outputVar);
            return [outputVar];
        } else {
            // Multi-output (branch) node
            const branchOp = node as BranchOp;
            const outputs: string[] = [];
            
            for (let i = 0; i < branchOp.outShapes.length; i++) {
                const outputVar = newVar();
                outputs.push(outputVar);
                nodeVars.set(`${node.id}_${i}`, outputVar);
            }
            
            return outputs;
        }
    }

    private _getNodeOperation(node: GraphNode): string {
        if (node instanceof Op) {
            return node.emitTorch();
        } else if (node instanceof MergeOp) {
            const tempCode = node.emitTorchFunctional(['temp1', 'temp2'], ['tempOut']);
            const match = tempCode.match(/= (.+)$/);
            return match ? match[1] : tempCode;
        } else if (node instanceof BranchOp) {
            const tempCode = node.emitTorchFunctional(['tempIn'], ['tempOut1', 'tempOut2']);
            const match = tempCode.match(/= (.+)$/);
            return match ? match[1] : tempCode;
        }
        
        throw new Error(`Unsupported node type: ${node.constructor.name}`);
    }

    private _formatNodeCode(node: GraphNode, inputs: string[], outputs: string[], operation: string): string {
        const inputStr = inputs.join(', ');
        const outputStr = outputs.join(', ');
        const nodeType = node.constructor.name;
        
        return `${outputStr} = ${operation}(${inputStr})  # ${nodeType}: ${node.id}\n`;
    }

    private _getSinkVar(sink: GraphNode, nodeVars: Map<string, string>): string {
        // Find the node that connects to this sink
        for (const [nodeId, node] of this._nodes.entries()) {
            if (node instanceof BranchOp) {
                for (let i = 0; i < node.nexts.length; i++) {
                    if (node.nexts[i] === sink) {
                        const key = `${nodeId}_${i}`;
                        const sinkVar = nodeVars.get(key);
                        if (sinkVar) return sinkVar;
                    }
                }
            } else if ((node instanceof Op || node instanceof MergeOp) && node.nexts[0] === sink) {
                const sinkVar = nodeVars.get(nodeId);
                if (sinkVar) return sinkVar;
            }
        }
        
        if (sink instanceof Tensor && sink.variableName) {
            return sink.variableName;
        }
        
        throw new Error(`No input variable found for sink ${sink.id}`);
    }

    // Helper methods for SSA code generation
    private _topologicalSort(): GraphNode[] {
        const visited = new Set<string>();
        const inProgress = new Set<string>();  // Cycle detection
        const result: GraphNode[] = [];
        
        const visit = (node: GraphNode) => {
            if (visited.has(node.id)) return;
            
            // Cycle detection
            if (inProgress.has(node.id)) {
                throw new Error(`Cycle detected involving node ${node.id}`);
            }
            
            inProgress.add(node.id);
            
            // Validate null entries in prevs (catch construction errors)
            for (let i = 0; i < node.prevs.length; i++) {
                const prev = node.prevs[i];
                if (prev === undefined) {
                    throw new Error(`Node ${node.id} has undefined entry at prevs[${i}] - graph construction error`);
                }
                if (prev !== null) {
                    visit(prev);
                }
            }
            
            inProgress.delete(node.id);
            visited.add(node.id);
            result.push(node);
        };
        
        // Sort nodes by id for deterministic order
        const sortedNodes = Array.from(this._nodes.values()).sort((a, b) => a.id.localeCompare(b.id));
        
        // Start from all nodes (sources will naturally be first due to no dependencies)
        for (const node of sortedNodes) {
            visit(node);
        }
        
        return result;
    }
      
    // Helper to safely get previous nodes regardless of node type
    private _getPrevNodes(node: GraphNode): GraphNode[] {
        return node.prevs.filter(p => p !== null);
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







