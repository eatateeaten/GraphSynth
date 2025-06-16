export function getLogicalComparisonOpCode(opType: string, input?: string): string {
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
    case 'div':
        return 'torch.div';
    default:
        throw new Error(`Unknown logical/comparison operation type: ${opType}`);
    }
} 