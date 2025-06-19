import { GraphNode } from './graph_node';
import { Tensor } from './tensor';
import { Graph } from './graph';

/**
 * Module represents a neural network module that can have multiple inputs and outputs.
 * It inherits from GraphNode and provides basic module functionality.
 */
export abstract class Module extends GraphNode {
    protected readonly _opType: string;
    protected _moduleName: string;
    private _graph: Graph;
    private _isFrozen: boolean = false;
    private _cachedTorchModule: string | null = null;
    private _cachedIR: string | null = null;

    constructor(
        id: string,
        opType: string,
        moduleName: string,
        params: Record<string, any> = {},
        graph?: Graph
    ) {
        super(id, params);
        this._opType = opType;
        this._moduleName = moduleName;
        this._graph = graph || new Graph();
    }

    // Operation getter
    get opType(): string { 
        return this._opType; 
    }

    // Module name getter
    get moduleName(): string {
        return this._moduleName;
    }

    // Module name setter
    set moduleName(name: string) {
        this._moduleName = name;
    }

    // Graph getter
    get graph(): Graph {
        return this._graph;
    }

    // Graph setter
    set graph(graph: Graph) {
        if (this._isFrozen) {
            throw new Error("Cannot modify graph: Module is frozen");
        }
        this._graph = graph;
        this._invalidateCache();
    }

    // Freeze control
    get isFrozen(): boolean {
        return this._isFrozen;
    }

    freeze(): void {
        if (!this._isFrozen) {
            // Validate the module before freezing
            this.validate();
            
            // Generate and cache code
            this._cachedTorchModule = this.toTorchModule();
            this._cachedIR = this.toIR();
            
            // Set the freeze state
            this._isFrozen = true;
            
            console.log(`Module "${this._moduleName}" frozen and code generated`);
        }
    }

    thaw(): void {
        this._isFrozen = false;
        this._invalidateCache();
    }

    // Get cached generated code (only available when locked)
    getCachedTorchModule(): string | null {
        return this._cachedTorchModule;
    }

    getCachedIR(): string | null {
        return this._cachedIR;
    }

    private _invalidateCache(): void {
        this._cachedTorchModule = null;
        this._cachedIR = null;
    }

    // Graph modification methods with freeze protection
    addNode(id: string, nodeType: string, params: Record<string, any>): void {
        if (this._isFrozen) {
            throw new Error("Cannot add node: Module is frozen");
        }
        this._graph.addNode(id, nodeType, params);
        this._invalidateCache();
    }

    connect(sourceId: string, sinkId: string, sourcePortIndex: number = 0, sinkPortIndex: number = 0): void {
        if (this._isFrozen) {
            throw new Error("Cannot connect nodes: Module is frozen");
        }
        this._graph.connect(sourceId, sinkId, sourcePortIndex, sinkPortIndex);
        this._invalidateCache();
    }

    disconnect(sourceId: string, sinkId: string, sourcePortIndex: number = 0, sinkPortIndex: number = 0): void {
        if (this._isFrozen) {
            throw new Error("Cannot disconnect nodes: Module is frozen");
        }
        this._graph.disconnect(sourceId, sinkId, sourcePortIndex, sinkPortIndex);
        this._invalidateCache();
    }

    // Read-only graph access methods (always allowed)
    getNode(id: string): GraphNode | undefined {
        return this._graph.getNode(id);
    }

    getAllNodes(): ReadonlyMap<string, GraphNode> {
        return this._graph.getAllNodes();
    }

    getEdges(): readonly any[] {
        return this._graph.getEdges();
    }

    // Tensor getters - dynamically computed from graph sources and sinks
    get inputs(): Tensor[] { 
        const sources = Array.from(this._graph.getSources());
        return sources.filter(node => node instanceof Tensor) as Tensor[];
    }
    
    get outputs(): Tensor[] { 
        const sinks = Array.from(this._graph.getSinks());
        return sinks.filter(node => node instanceof Tensor) as Tensor[];
    }

    // Shape getters - dynamically computed from graph tensor shapes
    get inShapes(): (number[] | null)[] {
        return this.inputs.map(tensor => tensor.inShapes[0] || null);
    }

    get outShapes(): (number[] | null)[] {
        return this.outputs.map(tensor => tensor.outShapes[0] || null);
    }

    /**
     * Abstract method to compute output shapes based on input shapes
     */
    protected abstract computeOutShape(): (number[] | null)[];

    /**
     * Abstract method to generate framework-specific code
     */
    abstract toTorchModule(): string;
    abstract toIR(): string;

    /**
     * Standard GraphNode interface methods for external connections
     */
    addPrev(prev: GraphNode, prevOutShape: number[], indexSelf?: number, indexPrev?: number): void {
        const actualOutShape = prev.outShapes[indexPrev || 0];
        if (actualOutShape && !GraphNode.shapeMatch(prevOutShape, actualOutShape)) {
            throw new Error(`Shape mismatch: expected ${prevOutShape}, got ${actualOutShape}`);
        }
        if (indexSelf === undefined) {
            this._prevs.push(prev);
            this._inShapes.push(prevOutShape);
        } else {
            if (indexSelf < 0 || indexSelf > this._prevs.length) {
                throw new Error(`Invalid input index: ${indexSelf}`);
            }
            this._prevs.splice(indexSelf, 0, prev);
            this._inShapes.splice(indexSelf, 0, prevOutShape);
        }
    }

    addNext(next: GraphNode, indexSelf?: number, indexNext?: number): void {
        if (indexSelf === undefined) {
            this._nexts.push(next);
        } else {
            if (indexSelf < 0 || indexSelf > this._nexts.length) {
                throw new Error(`Invalid output index: ${indexSelf}`);
            }
            this._nexts.splice(indexSelf, 0, next);
        }
    }

    deletePrev(indexSelf?: number): void {
        if (indexSelf === undefined) {
            this._prevs = [];
            this._inShapes = [];
        } else {
            if (indexSelf < 0 || indexSelf >= this._prevs.length) {
                throw new Error(`Invalid input index: ${indexSelf}`);
            }
            this._prevs.splice(indexSelf, 1);
            this._inShapes.splice(indexSelf, 1);
        }
    }

    deleteNext(indexSelf?: number): void {
        if (indexSelf === undefined) {
            this._nexts = [];
        } else {
            if (indexSelf < 0 || indexSelf >= this._nexts.length) {
                throw new Error(`Invalid output index: ${indexSelf}`);
            }
            this._nexts.splice(indexSelf, 1);
        }
    }

    // Backward compatibility getters/setters
    get prev(): GraphNode | null {
        return this._prevs.length > 0 ? this._prevs[0] : null;
    }

    set prev(node: GraphNode | null) {
        this._prevs = [];
        this._inShapes = [];
        
        if (node !== null) {
            const nodeOutShape = node.outShapes[0];
            if (!nodeOutShape) {
                throw new Error("Previous node must have a defined output shape");
            }
            this.addPrev(node, nodeOutShape);
        }
    }

    get next(): GraphNode | null {
        return this._nexts.length > 0 ? this._nexts[0] : null;
    }

    set next(node: GraphNode | null) {
        this._nexts = [];
        
        if (node !== null) {
            this.addNext(node);
        }
    }

    validate(): boolean {
        if (this._prevs.length > 0) {
            for (let i = 0; i < this._prevs.length; i++) {
                if (this._prevs[i] !== null && !this._inShapes[i]) {
                    throw new Error(`Missing input shape at index ${i}`);
                }
            }
        }

        const sources = this._graph.getSources();
        const sinks = this._graph.getSinks();
        
        if (sources.size === 0) {
            throw new Error("Internal graph must have at least one source");
        }
        
        if (sinks.size === 0) {
            throw new Error("Internal graph must have at least one sink");
        }

        return true;
    }
}

