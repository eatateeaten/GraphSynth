// Export shape types and utilities
export type { Shape } from './shape';

// Export graph
export { CheckerGraph } from './graph';

import { type UtilNodeParams } from './utils';

export type CheckerNodeType = keyof UtilNodeParams;

import { Tensor, TensorParams } from './utils';
import { Reshape, ReshapeParams } from './utils';
import { CheckerNode } from './node';

export type CheckerNodeConfig = { type: CheckerNodeType; params: Record<string, any> };

export function createCheckerNode(config: CheckerNodeConfig): CheckerNode<any> {
    switch (config.type) {
    case 'tensor':
        return new Tensor(config.params as TensorParams);
    case 'reshape':
        return new Reshape(config.params as ReshapeParams);
    default:
        throw new Error(`Unknown node type: ${config.type}`);
    }
}

export { CheckerNode };
