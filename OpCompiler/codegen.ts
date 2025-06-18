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
            const nodeIR = node.toIR();
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
     * Generates a complete PyTorch nn.Module class from the graph.
     * Creates both __init__ and forward methods with proper module instantiation.
     */
    public emitTorchModule(): string {
        this.validateGraph();
        
        const topoOrder = this.graph.getTopologicalOrder();
        const nodeToModuleName = new Map<string, string>();
        
        // Generate __init__ method
        const initCode = this._generateInitMethod(topoOrder, nodeToModuleName);
        
        // Generate forward method
        const forwardCode = this._generateForwardMethod(topoOrder, nodeToModuleName);
        
        // Combine into complete module
        return `import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratedModule(nn.Module):
    def __init__(self):
        super(GeneratedModule, self).__init__()
${initCode}
    
    def forward(self, ${this._getForwardInputs()}):
${forwardCode}
`;
    }

    private _generateInitMethod(topoOrder: GraphNode[], nodeToModuleName: Map<string, string>): string {
        let initCode = "";
        let moduleCounter = 0;
        
        for (const node of topoOrder) {
            // Skip source nodes (Tensors) - they don't need modules
            if (this.graph.getSources().has(node)) {
                continue;
            }
            
            // Generate module name
            const moduleName = `layer_${moduleCounter++}`;
            nodeToModuleName.set(node.id, moduleName);
            
            // Get the PyTorch module definition for this node
            // For init, we just need the module instantiation, not the forward call
            const moduleDefinition = node.toTorchModule();
            
            initCode += `        self.${moduleName} = ${moduleDefinition}  # ${node.id}\n`;
        }

        return initCode;
    }

    private _generateForwardMethod(topoOrder: GraphNode[], nodeToModuleName: Map<string, string>): string {
        let forwardCode = "";
        let varCounter = 0;
        const nodeToVar = new Map<string, string>();
        
        // Handle source nodes first
        const sources = Array.from(this.graph.getSources());
        for (let i = 0; i < sources.length; i++) {
            const inputVar = sources.length === 1 ? "x" : `x${i}`;
            nodeToVar.set(sources[i].id, inputVar);
        }
        
        for (const node of topoOrder) {
            // Skip source nodes - they're already handled as inputs
            if (this.graph.getSources().has(node)) {
                continue;
            }
            
            // Get input variables for this node
            const inputs = this._getNodeInputVars(node, nodeToVar);
            
            // Generate output variable
            const outputVar = `v${varCounter++}`;
            nodeToVar.set(node.id, outputVar);
            
            // Get module name
            const moduleName = nodeToModuleName.get(node.id);
            if (!moduleName) {
                throw new Error(`No module name found for node ${node.id}`);
            }
            
            // Generate forward pass code
            const inputStr = inputs.join(', ');
            forwardCode += `        ${outputVar} = self.${moduleName}(${inputStr})  # ${node.id}\n`;
        }
        
        // Generate return statement
        const sinks = Array.from(this.graph.getSinks());
        const returnVars = sinks.map(sink => {
            return nodeToVar.get(sink.id)!;
        });
        
        if (returnVars.length === 1) {
            forwardCode += `        return ${returnVars[0]}\n`;
        } else {
            forwardCode += `        return ${returnVars.join(', ')}\n`;
        }
        
        return forwardCode;
    }

    private _getForwardInputs(): string {
        const sources = Array.from(this.graph.getSources());
        if (sources.length === 1) {
            return "x";
        }
        return sources.map((_, i) => `x${i}`).join(', ');
    }

    private _getNodeInputVars(node: GraphNode, nodeToVar: Map<string, string>): string[] {
        const inputs: string[] = [];
        
        for (const prev of node.prevs) {
            if (prev) {
                const inputVar = nodeToVar.get(prev.id);
                if (inputVar) {
                    inputs.push(inputVar);
                }
            }
        }
        
        return inputs;
    }
}
