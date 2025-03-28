export const elementwiseOpMap: Record<string, string> = {
    "Add": "torch.add",
    "Multiply": "torch.mul",
    "Divide": "torch.div",
    "Subtract": "torch.sub",
    "Power": "torch.pow"
};

export function getElementwiseOpCode(opType: string): string {
    if (!(opType in elementwiseOpMap)) {
        throw new Error(`Unknown elementwise operation type: ${opType}`);
    }
    return elementwiseOpMap[opType];
} 