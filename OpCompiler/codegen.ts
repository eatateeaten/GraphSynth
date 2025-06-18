import { Graph, Edge } from './graph';
import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { BranchOp } from './branch_op';
import { MergeOp } from './merge_op';
import { 
    CycleError, 
    SourceNotTensorError, 
    SinkNotTensorError, 
    UnreachableNodeError 
} from './errors';

export class CodeGenerator {
    private graph: Graph;

    constructor(graph: Graph) {
        this.graph = graph;
    }

    /** Validates the graph before code generation */
    private validateGraph(): void {
        const sources = this.graph.getSources();
        const sinks = this.graph.getSinks();

        if (sources.size === 0) {
            throw new SourceNotTensorError("Graph has no source nodes");
        }

        if (sinks.size === 0) {
            throw new SinkNotTensorError("Graph has no sink nodes");
        }

        for (const source of sources) {
            if (!(source instanceof Tensor)) {
                throw new SourceNotTensorError(`Source node ${source.id} is not a Tensor (found ${source.constructor.name} instead)`);
            }
        }

        for (const sink of sinks) {
            if (!(sink instanceof Tensor)) {
                throw new SinkNotTensorError(`Sink node ${sink.id} is not a Tensor (found ${sink.constructor.name} instead)`);
            }
        }
        
        // Check for cycles and unreachable nodes
        try {
            this.graph.getTopologicalOrder();
        } catch (error) {
            if (error instanceof CycleError) {
                throw error;
            }
            throw new UnreachableNodeError(`Graph validation failed: ${error}`);
        }
    }

    /**
     * Generates intermediate representation (IR) for the entire graph.
     * Shows the structure and data flow of all nodes.
     */
    public emitIR(): string {
        this.validateGraph();
        
        const topoOrder = this.graph.getTopologicalOrder();
        let ir = "=== GRAPH IR ===\n";
        
        for (const node of topoOrder) {
            const nodeIR = node.emitIR();
            ir += `${node.id}: ${nodeIR}\n`;
        }
        
        // Add connection information
        ir += "\n=== CONNECTIONS ===\n";
        const edges = this.graph.getEdges();
        for (const edge of edges) {
            ir += `${edge.sourceId}[${edge.sourcePortIndex}] -> ${edge.sinkId}[${edge.sinkPortIndex}]\n`;
        }
        
        return ir;
    }

    /**
     * Generates complete SSA (Static Single Assignment) style PyTorch code from the graph DAG.
     * Each variable is assigned exactly once with automatic variable name generation.
     * Uses a depth-first traversal strategy to respect computation order.
     */
    public emitTorchFunctional(): string {
        this.validateGraph();
        
        let varCounter = 0;
        const newVar = () => `v${varCounter++}`;
        const nodeVars = new Map<string, string>();
        const topoOrder = this.graph.getTopologicalOrder();
        
        let code = "";
        
        for (const node of topoOrder) {
            const nodeCode = this._generateNodeCode(node, nodeVars, newVar);
            code += nodeCode;
        }
        
        // Generate return statement
        const sinks = this.graph.getSinks();
        const sinkVars = sinks.size > 0 
            ? Array.from(sinks).map(sink => this._getSinkVar(sink, nodeVars))
            : [];
            
        if (sinkVars.length > 0) {
            code += `return ${sinkVars.join(', ')}\n`;
        }
        
        return code;
    }

    private _generateNodeCode(node: GraphNode, nodeVars: Map<string, string>, newVar: () => string): string {
        const sources = this.graph.getSources();
        if (sources.has(node)) {
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
            const tempCode = node.emitTorchModule(['temp1', 'temp2'], ['tempOut']);
            const match = tempCode.match(/= (.+)$/);
            return match ? match[1] : tempCode;
        } else if (node instanceof BranchOp) {
            const tempCode = node.emitTorchModule(['tempIn'], ['tempOut1', 'tempOut2']);
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
        const nodes = this.graph.getAllNodes();
        for (const [nodeId, node] of nodes.entries()) {
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
      
    private _getPrevNodes(node: GraphNode): GraphNode[] {
        return node.prevs.filter(p => p !== null);
    }
}
