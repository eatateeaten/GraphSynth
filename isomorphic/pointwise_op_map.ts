/**
 * Returns framework-specific code for differentiable pointwise operations
 */
export function getDifferentiablePointWiseOpCode(opType: string, target: string = 'torch', input?: string): string {
    if (target.toLowerCase() === 'torch') {
        switch (opType.toLowerCase()) {
            case 'add':
                return 'torch.add';
            case 'mul':
                return 'torch.mul';
            case 'div':
                return 'torch.div';
            case 'pow':
                return 'torch.pow';
            case 'identity':
                return input ? input : 'x => x';
            default:
                throw new Error(`Unknown differentiable pointwise operation type: ${opType}`);
        }
    } else if (target.toLowerCase() === 'jax') {
        switch (opType.toLowerCase()) {
            case 'add':
                return 'jnp.add';
            case 'mul':
                return 'jnp.multiply';
            case 'div':
                return 'jnp.divide';
            case 'pow':
                return 'jnp.power';
            case 'identity':
                return input ? input : 'lambda x: x';
            default:
                throw new Error(`Unknown differentiable pointwise operation type: ${opType}`);
        }
    } else {
        throw new Error(`Unsupported target framework: ${target}`);
    }
}

/**
 * Returns framework-specific code for non-differentiable pointwise operations
 */
export function getNonDifferentiablePointWiseOpCode(opType: string, target: string = 'torch', input?: string): string {
    if (target.toLowerCase() === 'torch') {
        switch (opType.toLowerCase()) {
            case 'min':
                return 'torch.min';
            case 'max':
                return 'torch.max';
            case 'and':
                return 'torch.logical_and';
            case 'or':
                return 'torch.logical_or';
            case 'xor':
                return 'torch.logical_xor';
            case 'not':
                return 'torch.logical_not';
            case 'eq':
                return 'torch.eq';
            case 'ne':
                return 'torch.ne';
            case 'lt':
                return 'torch.lt';
            case 'le':
                return 'torch.le';
            case 'gt':
                return 'torch.gt';
            case 'ge':
                return 'torch.ge';
            default:
                throw new Error(`Unknown non-differentiable pointwise operation type: ${opType}`);
        }
    } else if (target.toLowerCase() === 'jax') {
        switch (opType.toLowerCase()) {
            case 'min':
                return 'jnp.minimum';
            case 'max':
                return 'jnp.maximum';
            case 'and':
                return 'jnp.logical_and';
            case 'or':
                return 'jnp.logical_or';
            case 'xor':
                return 'jnp.logical_xor';
            case 'not':
                return 'jnp.logical_not';
            case 'eq':
                return 'jnp.equal';
            case 'ne':
                return 'jnp.not_equal';
            case 'lt':
                return 'jnp.less';
            case 'le':
                return 'jnp.less_equal';
            case 'gt':
                return 'jnp.greater';
            case 'ge':
                return 'jnp.greater_equal';
            default:
                throw new Error(`Unknown non-differentiable pointwise operation type: ${opType}`);
        }
    } else {
        throw new Error(`Unsupported target framework: ${target}`);
    }
} 