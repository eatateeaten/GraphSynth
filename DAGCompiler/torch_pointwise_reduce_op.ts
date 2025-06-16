// Add getElementwiseOpCode to ensure backward compatibility
export function getPointWiseReduceOpCode(opType: string, input?: string): string {
    switch (opType.toLowerCase()) {
    case 'add':
        return 'torch.add'; 
    case 'mul':
        return 'torch.mul';
    case 'pow':
        return 'torch.pow';
    case 'identity':
        return input ? input : 'x => x'; // Return input unchanged
    default:
        throw new Error(`Unknown differentiable elementwise operation type: ${opType}`);
    }
}

/**
 * Returns PyTorch code for non-differentiable operations that are associative and commutative
 */
export function getNonDifferentiableAssociativeCommutativeOpCode(opType: string, input?: string): string {
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
}