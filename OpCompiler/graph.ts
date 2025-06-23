import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Op } from './op';
import { Split, Copy } from './branch_op';
import { PointwiseOp, DotOp, CrossOp } from './merge_op';
import { Concat, PointwiseReduce } from './reduce_op';
import { NodeType } from "./types";
import { ModuleDB } from '../moduledb';
import { 
    ShapeMatchError, 
    ParamError, 
    CycleError, 
} from './errors';

/// Node factory lookup by NodeType
const nodeFactories: Record<NodeType, (id: string, moduleName: string, params: Record<string, any>) => GraphNode> = {
    [NodeType.TENSOR]: (id, _, params) => new Tensor(id, params.shape, params.variableName),
    [NodeType.OP]: (id, moduleName, params) => new Op(id, moduleName, params),
    [NodeType.SPLIT]: (id, _, params) => new Split(id, params.dim, params.sections, params),
    [NodeType.COPY]: (id, _, params) => new Copy(id, params.copies, params),
    [NodeType.CONCAT]: (id, _, params) => new Concat(id, params.dim, params.numberOfMerges, params),
    [NodeType.POINTWISE_REDUCE]: (id, moduleName, params) => new PointwiseReduce(id, moduleName, params.numberOfMerges, params),
    [NodeType.POINTWISE_OP]: (id, moduleName, params) => new PointwiseOp(id, moduleName, params),
    [NodeType.DOT_OP]: (id, _, params) => new DotOp(id, params),
    [NodeType.CROSS_OP]: (id, _, params) => new CrossOp(id, params),
}; //TODO Figure out a way to catch errors 

/**
 * Interface defining a connection edge between two nodes in the graph
 */
export interface Edge {
    edgeId: string;     // Unique identifier for the edge
    sourceId: string;
    sinkId: string;
    sourcePortIndex?: number;
    sinkPortIndex?: number;
}

export class Graph {
    private _nodes: Map<string, GraphNode>;
    private _edges: Edge[]; // Track all connections for easier disconnection

    constructor() {
        this._nodes = new Map();
        this._edges = [];
    }

    getNode(id: string): GraphNode | undefined {
        return this._nodes.get(id);
    }

    getAllNodes(): ReadonlyMap<string, GraphNode> {
        return this._nodes;
    }

    getSources(): ReadonlySet<GraphNode> {
        const sources = new Set<GraphNode>();
        for (const node of this._nodes.values()) {
            if (!GraphNode.hasInputs(node)) {
                sources.add(node);
            }
        }
        return sources;
    }

    getSinks(): ReadonlySet<GraphNode> {
        const sinks = new Set<GraphNode>();
        for (const node of this._nodes.values()) {
            if (!GraphNode.hasOutputs(node)) {
                sinks.add(node);
            }
        }
        return sinks;
    }

    getEdges(): readonly Edge[] {
        return this._edges;
    }

        /** Adds a node to the graph */
    addNode(id: string, moduleName: string, params: Record<string, any>): void {
        // Get module definition from ModuleDB
        const module = ModuleDB.get(moduleName);
        if (!module) {
            throw new ParamError(`Unknown module: ${moduleName}`);
        }

        // Get the node type from the module
        const nodeType = module.moduleType;
        if (!nodeType) {
            throw new ParamError(`Module ${moduleName} has no nodeType defined`);
        }

        // Create node using factory
        const factory = nodeFactories[nodeType];
        if (!factory) {
            throw new ParamError(`Unsupported node type: ${nodeType}`);
        }

        const node = factory(id, moduleName, params);
        this._nodes.set(id, node);
    }

    /** Removes a node from the graph */
    removeNode(nodeId: string): void {
        throw new Error("removeNode is not implemented");
    }

    // /** Adds a node to the graph */
    // addNode(id: string, nodeType: string, params: Record<string, any>): void {
    //     if(!isNodeType(nodeType))
    //         throw new ParamError(`Unknown GraphNode type: ${nodeType}`);

    //     const factory = moduleFromParams[nodeType];

    //     const node = factory(id as NodeType, params);
    //     this._nodes.set(id, node);
    // }

    // /** Removes a node from the graph */
    // removeNode(nodeId: string): void {
    //     throw new Error("removeNode is not implemented");
    // }

    /** Connects two nodes and updates their I/O shapes.
     * Throws if shapes don't get matched or validated. */
    connect(sourceId: string, sinkId: string, sourcePortIndex: number, sinkPortIndex: number): void {
        const source = this._nodes.get(sourceId);
        if (!source) throw new Error(`Source node with id ${sourceId} does not exist in graph`);
        const sink = this._nodes.get(sinkId);
        if (!sink) throw new Error(`Sink node with id ${sinkId} does not exist in graph`);

        // Validate source port
        const sourceShape = source.outShapes[sourcePortIndex];
        if (!sourceShape) {
            throw new ShapeMatchError(`Cannot connect from ${source.constructor.name} with id ${source.id}: output shape is undefined`);
        }

        // Validate sink port
        const sinkShape = sink.inShapes[sinkPortIndex];
        if (sinkShape === null && !GraphNode.inShapeInferred(sink)) {
            throw new Error(`Node with id ${sink.id} must have defined input shape`);
        }

        // Shape compatibility check
        if (sinkShape !== null && !GraphNode.shapeMatch(sourceShape, sinkShape)) {
            throw new ShapeMatchError(`Shape mismatch: Cannot connect ${source.constructor.name} with output shape [${sourceShape}] to ${sink.constructor.name} with input shape [${sinkShape}]`);
        }

        // Establish bidirectional references
        sink.addPrev(source, sourceShape, sinkPortIndex, sourcePortIndex);
        source.addNext(sink, sourcePortIndex, sinkPortIndex);

        // Add the connection to the edge list with a unique ID 
        this._edges.push({
            edgeId: this._generateUUID(),
            sourceId: sourceId,
            sinkId: sinkId,
            sourcePortIndex: sourcePortIndex,
            sinkPortIndex: sinkPortIndex
        });
    }
    
    /** Disconnect two nodes */
    disconnect(sourceId: string, sinkId: string, sourcePortIndex: number, sinkPortIndex: number): void {        
        const source = this._nodes.get(sourceId);
        const sink = this._nodes.get(sinkId);

        if (!source) throw new Error(`No source with ID ${sourceId}`);
        if (!sink) throw new Error(`No sink with ID ${sinkId}`);

        // Remove bidirectional node connections
        sink.deletePrev(sinkPortIndex);
        source.deleteNext(sourcePortIndex);

        // Remove the corresponding edge from the edge list 
        this._edges = this._edges.filter(edge => 
            !(edge.sourceId === sourceId && edge.sinkId === sinkId && 
              edge.sourcePortIndex === sourcePortIndex && edge.sinkPortIndex === sinkPortIndex)
        );
    }

    /** General-purpose BFS traversal with visitor pattern.
     * Visits nodes in breadth-first order starting from nodes with no incoming edges. */
    traverseBFS(visitor: (node: GraphNode) => void): void {
        const inDegree = new Map<string, number>();
        for (const [id, node] of this._nodes) {
            inDegree.set(id, node.prevs.filter(p => p !== null).length);
        }
        
        const queue: GraphNode[] = [];
        for (const [id, degree] of inDegree) {
            if (degree === 0) {
                queue.push(this._nodes.get(id)!);
            }
        }
        
        while (queue.length > 0) {
            const node = queue.shift()!;
            visitor(node);
            
            for (const nextNode of node.nexts) {
                if (!nextNode) continue;
                
                const newDegree = inDegree.get(nextNode.id)! - 1;
                inDegree.set(nextNode.id, newDegree);
                
                if (newDegree === 0) {
                    queue.push(nextNode);
                }
            }
        }
    }

    /** Returns nodes in topological order.
     * Throws CycleError if the graph contains cycles. */
    getTopologicalOrder(): GraphNode[] {
        const result: GraphNode[] = [];
        const visited = new Set<string>();

        this.traverseBFS(node => {
            result.push(node);
            visited.add(node.id);
        });

        if (visited.size !== this._nodes.size) {
            const unprocessed = Array.from(this._nodes.keys())
                .filter(id => !visited.has(id));
            throw new CycleError(`Graph contains cycles involving nodes: ${unprocessed.join(', ')}`);
        }
        
        return result;
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
}
