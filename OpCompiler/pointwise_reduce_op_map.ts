/**
 * Returns framework-specific code for differentiable operations that are associative and commutative
 */
export function getPointWiseReduceOpCode(opType: string, target: string = 'torch', input?: string): string {
    if (target.toLowerCase() === 'torch') {
        switch (opType.toLowerCase()) {
        case 'add':
            return 'torch.add'; 
        case 'mul':
            return 'torch.mul';
        case 'pow':
            return 'torch.pow';
        case 'identity':
            return input ? input : 'x => x';
        default:
            throw new Error(`Unknown differentiable elementwise operation type: ${opType}`);
        }
    } else {
        throw new Error(`Unsupported target framework: ${target}`);
    }
}

/**
 * Returns framework-specific code for non-differentiable operations that are associative and commutative
 */
export function getNonDifferentiableReduceOpCode(opType: string, target: string = 'torch', input?: string): string {
    if (target.toLowerCase() === 'torch') {
        switch (opType.toLowerCase()) {
        case 'and':
            return 'torch.logical_and';
        case 'or':
            return 'torch.logical_or';
        case 'xor':
            return 'torch.logical_xor';
        default:
            throw new Error(`Unknown non-differentiable associative and commutative operation type: ${opType}`);
        }
    } else {
        throw new Error(`Unsupported target framework: ${target}`);
    }
}
