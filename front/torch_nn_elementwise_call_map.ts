export const elementwiseOpMap: Record<string, string> = {
    "Add": "torch.add",
    "Multiply": "torch.mul",
    "Divide": "torch.div",
    "Subtract": "torch.sub",
    "Power": "torch.pow"
};

export function getElementwiseOpCode(opType: string): string {
    switch (opType.toLowerCase()) {
        case 'add':
            return 'torch.add';
        case 'sub':
            return 'torch.sub';
        case 'mul':
            return 'torch.mul';
        case 'div':
            return 'torch.div';
        case 'pow':
            return 'torch.pow';
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
            throw new Error(`Unknown elementwise operation type: ${opType}`);
    }
    return elementwiseOpMap[opType];
} 