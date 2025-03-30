import { MergeOp } from '../front/merge_op';
import { BranchOp } from '../front/branch_op';
import { GraphNode } from '../front/graph';
import { Tensor } from '../front/tensor';
import { Op } from '../front/op';

/**
 * TestMergeOp - A simple implementation of MergeOp for testing
 * Just passes through the first input in to_torch_functional
 */
export class TestMergeOp extends MergeOp {
    constructor(
        id: string,
        inShapes: number[][],
        target: string
    ) {
        super(id, inShapes, target, 'TestMerge', {});
    }

    protected computeOutShape(): number[] {
        // Just use the first input shape as the output shape
        if (this._inShapes.length === 0) {
            throw new Error("TestMergeOp requires at least one input shape");
        }
        return [...this._inShapes[0]];
    }

    to_torch_functional(inputs: string[]): string {
        // Simply pass through the first input
        if (inputs.length === 0) {
            throw new Error("TestMergeOp requires at least one input");
        }
        return `${inputs[0]} = ${inputs[0]}`;
    }
}

/**
 * TestBranchOp - A simple implementation of BranchOp for testing
 * Copies the input to multiple outputs in to_torch_functional
 */
export class TestBranchOp extends BranchOp {
    constructor(
        id: string,
        inShape: number[],
        target: string,
        numOutputs: number = 2
    ) {
        super(id, inShape, target, 'TestBranch', { numOutputs });
    }

    protected computeOutShapes(): number[][] {
        // Create identical output shapes for each output
        const numOutputs = this._params.numOutputs || 2;
        const outShapes: number[][] = [];
        
        for (let i = 0; i < numOutputs; i++) {
            outShapes.push([...this._inShape]);
        }
        
        return outShapes;
    }

    to_torch_functional(inputs: string[]): string {
        // Copy the input to each output
        if (inputs.length === 0) {
            throw new Error("TestBranchOp requires an input");
        }

        const numOutputs = this._params.numOutputs || 2;
        
        if (numOutputs === 1) {
            return `${inputs[0]} = ${inputs[0]}`;
        }
        
        const outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            outputs.push(`${inputs[0]}_${i} = ${inputs[0]}`);
        }
        
        return outputs.join('\n');
    }
}

/**
 * TestGraph - A copy of the original Graph class that works with
 * Tensor, Op, TestMergeOp and TestBranchOp
 */
export class TestGraph {
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
        // Validate graph before generating code
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
        if (this._sources.size === 0) {
            throw new Error("Graph has no source nodes");
        }
        
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
        
        // Check that all sinks are reachable from sources
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
            } else if (node instanceof TestBranchOp) {
                for (const nextNode of node._nexts) {
                    if (nextNode) {
                        queue.push(nextNode);
                    }
                }
            } else if (node instanceof TestMergeOp) {
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
        } else if (node instanceof TestBranchOp) {
            for (const nextNode of node._nexts) {
                if (nextNode) {
                    this._dfsCheckCycle(nextNode, visiting, visited);
                }
            }
        } else if (node instanceof TestMergeOp) {
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
        } else if (node instanceof TestBranchOp) {
            if (node.prev) {
                const prevVar = this._ensureNodeProcessed(node.prev, processedNodes, varCounter);
                inputVars.push(prevVar);
            }
        } else if (node instanceof TestMergeOp) {
            // Process all inputs for TestMergeOp
            for (const prevNode of node._prevs) {
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
        } else if (node instanceof TestBranchOp) {
            for (const nextNode of node._nexts) {
                if (nextNode && !processedNodes.has(nextNode.id)) {
                    code += this._processNode(nextNode, processedNodes, varCounter);
                }
            }
        } else if (node instanceof TestMergeOp) {
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
        if (node instanceof Tensor || node instanceof Op || node instanceof TestBranchOp) {
            if (node.prev === null) {
                this._sources.add(node);
            } else {
                this._sources.delete(node);
            }
        } else if (node instanceof TestMergeOp) {
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
        } else if (node instanceof TestBranchOp) {
            if (node._nexts.every(n => !n)) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        } else if (node instanceof TestMergeOp) {
            if (node.next === null) {
                this._sinks.add(node);
            } else {
                this._sinks.delete(node);
            }
        }
    }
} 