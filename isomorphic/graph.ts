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

    abstract connectSource(prev: GraphNode, indexSelf?: number, indexPrev?: number): void;
    abstract connectSink(next: GraphNode, indexSelf?: number, indexNext?: number): void;
    abstract disconnectSource(indexSelf?: number): void;
    abstract disconnectSink(indexSelf?: number): void;
    abstract to_torch_functional(inputs: string[]): string;

    static validateIndex(index: number, length: number, context: string): number {
        if (index < 0 || index >= length) {
            throw new Error(`${context}: Index ${index} is out of bounds for length ${length}`);
        }
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
}

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _sources: Set<GraphNode>;
    private _sinks: Set<GraphNode>;

    constructor() {
        this._nodes = new Map();
        this._sources = new Set();
        this._sinks = new Set();
    }

    addNode(node: GraphNode): void {
        if (this._nodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} already exists in graph`);
        }
        
        // Add node to the graph
        this._nodes.set(node.id, node);
        
        // When a node is first added, it's both a source and a sink
        // since it has no connections yet
        this._sources.add(node);
        this._sinks.add(node);
    }

    removeNode(node: GraphNode): void {
        if (!this._nodes.has(node.id)) {
            throw new Error(`Node with id ${node.id} does not exist in graph`);
        }

        // Disconnect all connections
        node.disconnectSource();
        node.disconnectSink();

        // Remove from collections
        this._nodes.delete(node.id);
        this._sources.delete(node);
        this._sinks.delete(node);
    }

    connect(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        if (!this._nodes.has(sink.id)) {
            throw new Error(`Sink node with id ${sink.id} does not exist in graph`);
        }

        sink.connectSource(source, sinkIndex, sourceIndex);
        source.connectSink(sink, sourceIndex, sinkIndex);

        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
    }

    disconnect(source: GraphNode, sink: GraphNode, sourceIndex?: number, sinkIndex?: number): void {
        if (!this._nodes.has(source.id)) {
            throw new Error(`Source node with id ${source.id} does not exist in graph`);
        }
        if (!this._nodes.has(sink.id)) {
            throw new Error(`Sink node with id ${sink.id} does not exist in graph`);
        }

        sink.disconnectSource(sinkIndex);
        source.disconnectSink(sourceIndex);

        this._refreshNodeSinkSourceStatus(source);
        this._refreshNodeSinkSourceStatus(sink);
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
        if (node instanceof Tensor || node instanceof Op) {
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
        } else if (node instanceof MergeOp) {
            if (node.next === null) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        }
    }
}







