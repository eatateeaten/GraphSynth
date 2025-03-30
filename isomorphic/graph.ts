// torch.abs(input)       # Absolute value
// torch.sqrt(input)      # Square root
// torch.square(input)    # Square
// torch.exp(input)       # Exponential
// torch.log(input)       # Natural logarithm
// torch.sin(input)
// torch.cos(input)
// torch.tan(input)
// torch.asin(input)
// torch.acos(input)
// torch.atan(inpu

//The Graph  
//Merge Node 
//Shape check for Merge and Branch 
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp } from './branch_op';
import { MergeOp, Concat } from './merge_op';
import { Split } from './branch_op';

export { Tensor, Op, Concat, Split, BranchOp, MergeOp };

export abstract class GraphNode {
    protected readonly _id: string;
    protected readonly _target: string;
    constructor(id: string, target: string) {
        // Validate UUID format
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        if (!uuidRegex.test(id)) {
            throw new Error(`Invalid UUID format: ${id}`);
        }
        this._id = id;
        this._target = target;
    }

    get id(): string { return this._id; }
    get target(): string { return this._target; }

    abstract get prev(): GraphNode | null;
    abstract set prev(node: GraphNode | null);
    abstract get next(): GraphNode | null;
    abstract set next(node: GraphNode | null);

    abstract addPrev(prev: GraphNode, indexSelf?: number, indexPrev?: number): void;
    abstract addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract deletePrev(indexSelf?: number): void;
    abstract deleteNext(indexSelf?: number): void;
    abstract to_torch_functional(inputs: string[]): string;

    static checkIndexInBound(index: number, length: number, context: string): number {
        if (index < 0 || index >= length) {throw new Error(`${context}: Index ${index} is out of bounds for length ${length}`);}
        return index;
    }

    static shapeMatch(shape1: number[], shape2: number[]): boolean {
        if (shape1.length !== shape2.length) {
            return false;
        }
        for (let i = 0; i < shape1.length; i++) {
            if (shape1[i] !== shape2[i]) {
                return false;
            }
        }
        return true;
    }

    /* 
    // Future enhancement: Support for shape matching with broadcasting rules
    static smartShapeMatch(shape1: number[], shape2: number[]): boolean {
        // Simple exact matching first
        if (shape1.length === shape2.length && shape1.every((dim, i) => dim === shape2[i])) {
            return true;
        }
        
        // Broadcasting support (compatible with NumPy/PyTorch rules)
        const len1 = shape1.length;
        const len2 = shape2.length;
        const maxLen = Math.max(len1, len2);
        
        // Check dimensions from right to left
        for (let i = 1; i <= maxLen; i++) {
            const dim1 = i <= len1 ? shape1[len1 - i] : 1;
            const dim2 = i <= len2 ? shape2[len2 - i] : 1;
            
            // Dimensions must be equal or one of them must be 1
            if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
                return false;
            }
        }
        
        return true;
    }
    */
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
                node = new Tensor(id, params.shape, target);
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
     */
    addPendingNode<T extends GraphNode>(node: T): PendingNode<T> {
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
     * Creates and adds a node to the pending collection using the specified type and parameters
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
     * Removes a node from the pending collection
     */
    removePendingNode(nodeId: string): void {
        if (!this._pendingNodes.has(nodeId)) {
            throw new Error(`Node with id ${nodeId} is not a pending node`);
        }
        
        this._pendingNodes.delete(nodeId);
    }

    /**
     * Removes a node from the main graph.
     * Note: The node must not have any active connections.
     */
    removeNode(nodeId: string): void {
        const node = this._nodes.get(nodeId);
        if (!node) {throw new Error(`Node with id ${nodeId} does not exist in graph`);}
        // Check if the node has any connections using a helper function
        const hasConnections = this.nodeHasConnections(node);
        if (hasConnections) {throw new Error(`Cannot remove node ${nodeId}: node has active connections`);}
        // Remove from collections
        this._nodes.delete(nodeId);
        this._sources.delete(node);
        this._sinks.delete(node);
    }
    
    private nodeHasConnections(node: GraphNode): boolean {
        if (node instanceof Tensor || node instanceof Op || node instanceof BranchOp) {if (node.prev !== null) return true;}
        if (node instanceof Tensor || node instanceof Op || node instanceof MergeOp) {if (node.next !== null) return true; }
        if (node instanceof BranchOp && node._nexts.some(n => n !== null)) return true;
        if (node instanceof MergeOp && node._prevs.some(p => p !== null)) return true;
        return false;
    }

    /**
     * Connects two nodes. The source must be in the main graph, but the sink can be pending.
     */
    connect(source: GraphNode, sink: GraphNode | PendingNode<GraphNode>, sourceIndex?: number, sinkIndex?: number): void {
        const actualSink: GraphNode = sink instanceof PendingNode ? sink.unwrap() : sink;
        
        // Source node must exist in the main graph
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        
        // If sink is not pending, check if it exists in the main graph
        if (!(sink instanceof PendingNode) && !this._nodes.has(actualSink.id)) {
            throw new Error(`Sink node with id ${actualSink.id} does not exist in graph or pending nodes`);
        }

        // Validate connection endpoints
        sourceIndex = this.checkConnectionSource(source, sourceIndex);
        sinkIndex = this.checkConnectionSink(actualSink, sinkIndex);
       
        // Validate shape compatibility
        const sourceOutShape = this.getSourceOutShape(source, sourceIndex);
        const sinkInShape = this.getSinkInShape(actualSink, sinkIndex);

        // Check shape compatibility
        if (!GraphNode.shapeMatch(sourceOutShape, sinkInShape)) {
            throw new Error(`Shape mismatch: Cannot connect ${source.constructor.name} with output shape [${sourceOutShape}] to ${actualSink.constructor.name} with input shape [${sinkInShape}]`);
        }

        // Establish bidirectional connections
        // Let each node handle its own connection logic
        actualSink.addPrev(source, sinkIndex, sourceIndex);
        source.addNext(actualSink, sourceIndex, sinkIndex);

        // Only if the connection was successful and sink was pending, move it to the main graph
        if (sink instanceof PendingNode) {
            this._pendingNodes.delete(actualSink.id);
            this._nodes.set(actualSink.id, actualSink);
        }

        // Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(actualSink);
    }
    
    private checkConnectionSource(source: GraphNode, sourceIndex?: number): number | undefined {
        // For BranchOp, validate the index
        if (source instanceof BranchOp) {
            if (sourceIndex === undefined) {
                throw new Error("When connecting from a BranchOp, an output index must be specified");
            }
            return GraphNode.checkIndexInBound(sourceIndex, source.outShape.length, "connect (BranchOp output)");
        }
        return sourceIndex;
    }
    private getSourceOutShape(source: GraphNode, sourceIndex?: number): number[] {
        // Get source's output shape
        let sourceOutShape: number[];

        if (source instanceof Tensor || source instanceof Op || source instanceof MergeOp) {
            if (source.outShape === null) {
                throw new Error(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
            }
            sourceOutShape = source.outShape;
        } else if (source instanceof BranchOp) {
            const branchOutShape = source.outShape[sourceIndex!];
            if (branchOutShape === null || branchOutShape === undefined) {
                throw new Error(`Cannot connect from BranchOp with id ${source.id} at output ${sourceIndex}: output shape is undefined`);
            }
            sourceOutShape = branchOutShape;
        } else {
            throw new Error(`Unknown source node of type ${source.constructor.name}`);
        }

        return sourceOutShape;
    }

    private checkConnectionSink(sink: GraphNode, sinkIndex?: number): number | undefined {
        // For MergeOp, validate the index
        if (sink instanceof MergeOp) {
            if (sinkIndex === undefined) {
                throw new Error("When connecting to a MergeOp, an input index must be specified");
            }
            return GraphNode.checkIndexInBound(sinkIndex, sink.inShape.length, "connect (MergeOp input)");
        }
        return sinkIndex;
    }

    private getSinkInShape(sink: GraphNode, sinkIndex?: number): number[] {
        // Get sink's input shape
        if (sink instanceof Tensor || sink instanceof Op || sink instanceof BranchOp) {
            return sink.inShape as number[];
        } else if (sink instanceof MergeOp) {
            return sink.inShape[sinkIndex!];
        } else {
            throw new Error(`Unknown sink node of type ${sink.constructor.name}`);
        }
    }


    disconnect(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        if (!this._nodes.has(sink.id)) {
            throw new Error(`Sink node with id ${sink.id} does not exist in graph`);
        }

        // Step 1: Validate indices for special node types
        sourceIndex = this.validateDisconnectionIndices(source, sink, sourceIndex, sinkIndex);
        sinkIndex = this.validateDisconnectionIndices(sink, source, sinkIndex, sourceIndex);

        // Step 2: Verify connections exist
        this.verifyConnectionExists(source, sink, sourceIndex, sinkIndex);

        // Step 3: Break connections using node methods
        sink.deletePrev(sinkIndex);
        source.deleteNext(sourceIndex);

        // Step 4: Update graph status
        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    private validateDisconnectionIndices(node: GraphNode, otherNode: GraphNode, index?: number, otherIndex?: number): number | undefined {
        if (node instanceof BranchOp && index !== undefined) {
            return GraphNode.checkIndexInBound(index, node.outShape.length, "disconnect (BranchOp output)");
        }
        if (node instanceof MergeOp && index !== undefined) {
            return GraphNode.checkIndexInBound(index, node.inShape.length, "disconnect (MergeOp input)");
        }
        return index;
    }

    private verifyConnectionExists(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
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
        // Check if node is a source (no incoming connections)
        if (node instanceof Tensor || node instanceof Op || node instanceof BranchOp) {
            if (node.prev === null) {
                this._sources.add(node);
            } else {
                this._sources.delete(node);
            }
        } else if (node instanceof MergeOp) {
            if (node._prevs.every(p => !p)) {
                this._sources.add(node);
            } else {
                this._sources.delete(node);
            }
        }

        // Check if node is a sink (no outgoing connections)
        if (node instanceof Tensor || node instanceof Op || node instanceof MergeOp) {
            if (node.next === null) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        } else if (node instanceof BranchOp) {
            if (node._nexts.every(n => !n)) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        }
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

    to_torch(): string {
        // Validate the graph before generating code
        this.validate_graph();
        
        // Dictionary to track processed nodes and their output variable names
        const processedNodes = new Map<string, string>();
        const varCounter = { value: 0 };
        
        let code = "";
        
        // Process all source nodes first
        const sourceNodes = Array.from(this._sources);
        for (const sourceNode of sourceNodes) {
            code += this._processNode(sourceNode, processedNodes, varCounter);
        }
        
        return code;
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
    
    private _processNode(node: GraphNode, processedNodes: Map<string, string>, varCounter: { value: number }): string {
        // If already processed, return empty string
        if (processedNodes.has(node.id)) {
            return "";
        }
        
        let code = "";
        let inputVars: string[] = [];
        
        // Process input nodes first and collect their output variable names
        if (node instanceof Tensor) {
            // For tensor, no dependencies to process
            if (node.prev) {
                const prevVar = this._ensureNodeProcessed(node.prev, processedNodes, varCounter);
                inputVars.push(prevVar);
            } else {
                // Source tensor gets its own variable
                inputVars = [];
            }
        } else if (node instanceof Op) {
            if (node.prev) {
                const prevVar = this._ensureNodeProcessed(node.prev, processedNodes, varCounter);
                inputVars.push(prevVar);
            }
        } else if (node instanceof BranchOp) {
            if (node.prev) {
                const prevVar = this._ensureNodeProcessed(node.prev, processedNodes, varCounter);
                inputVars.push(prevVar);
            }
        } else if (node instanceof MergeOp) {
            // Process all inputs for MergeOp
            for (const prevNode of (node as MergeOp)._prevs) {
                if (prevNode) {
                    const prevVar = this._ensureNodeProcessed(prevNode, processedNodes, varCounter);
                    inputVars.push(prevVar);
                }
            }
        }
        
        // Generate variable name for this node
        const outputVar = `var_${varCounter.value++}`;
        processedNodes.set(node.id, outputVar);
        
        // Generate code for this node
        let nodeFunctionalCode = node.to_torch_functional(inputVars).trim();
        
        // If the code doesn't assign to our output variable, wrap it
        if (!nodeFunctionalCode.startsWith(`${outputVar} =`)) {
            // For tensor inputs, use their ID as variable
            if (node instanceof Tensor && inputVars.length === 0) {
                nodeFunctionalCode = `${outputVar} = ${node.id}  # Input tensor`;
            } else {
                nodeFunctionalCode = `${outputVar} = ${nodeFunctionalCode}`;
            }
        }
        
        code += nodeFunctionalCode + "\n";
        
        // Process next nodes
        if (node instanceof Tensor || node instanceof Op) {
            if (node.next && !processedNodes.has(node.next.id)) {
                code += this._processNode(node.next, processedNodes, varCounter);
            }
        } else if (node instanceof BranchOp) {
            for (const nextNode of (node as BranchOp)._nexts) {
                if (nextNode && !processedNodes.has(nextNode.id)) {
                    code += this._processNode(nextNode, processedNodes, varCounter);
                }
            }
        } else if (node instanceof MergeOp) {
            if (node.next && !processedNodes.has(node.next.id)) {
                code += this._processNode(node.next, processedNodes, varCounter);
            }
        }
        
        return code;
    }
    
    private _ensureNodeProcessed(node: GraphNode, processedNodes: Map<string, string>, varCounter: { value: number }): string {
        if (!processedNodes.has(node.id)) {
            this._processNode(node, processedNodes, varCounter);
        }
        return processedNodes.get(node.id)!;
    }
}







