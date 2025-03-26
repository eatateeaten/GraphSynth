export const pointwiseOpMap: Record<string, string> = {
    "Add": "torch.add",
    "Multiply": "torch.mul",
    "Divide": "torch.div",
    "Subtract": "torch.sub",
    "Power": "torch.pow"
};

export function getPointwiseOpCode(opType: string): string {
    if (!(opType in pointwiseOpMap)) {
        throw new Error(`Unknown pointwise operation type: ${opType}`);
    }
    return pointwiseOpMap[opType];
} 