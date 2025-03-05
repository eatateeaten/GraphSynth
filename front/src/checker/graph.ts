// src/checker/graph.ts
import { CheckerNode, NodeParams } from './node';

export class CheckerGraph {
    private nodes = new Map<string, CheckerNode<any>>();

    addNode<T extends NodeParams>(id: string, node: CheckerNode<T>): void {
        this.nodes.set(id, node);
    }

    getNode(id: string): CheckerNode<any> | undefined {
        return this.nodes.get(id);
    }

    connect(sourceId: string, targetId: string): void {
        const source = this.nodes.get(sourceId);
        const target = this.nodes.get(targetId);
        
        if (!source || !target) {
            throw new Error("Node not found");
        }

        source.connect_to(target);
    }

    deleteNode(id: string): void {
        const node = this.nodes.get(id);
        if (!node) return;

        if (node.in_node) {
            node.in_node.out_node = null;
        }
        if (node.out_node) {
            node.out_node.in_node = null;
        }

        this.nodes.delete(id);
    }
}
