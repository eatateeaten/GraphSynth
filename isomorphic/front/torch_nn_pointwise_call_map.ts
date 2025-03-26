export const pointwiseOpMap: Record<string, string> = {
    "Add": "+",
    "Multiply": "*",
    "Divide": "/",
    "Subtract": "-",
    "Maximum": "torch.maximum",
    "Minimum": "torch.minimum",
    "Power": "**",
    "Equal": "==",
    "NotEqual": "!=",
    "Greater": ">",
    "GreaterEqual": ">=",
    "Less": "<",
    "LessEqual": "<=",
    "LogicalAnd": "&",
    "LogicalOr": "|",
    "LogicalXor": "^"
};

export function getPointwiseOpCode(opType: string): string {
    if (!(opType in pointwiseOpMap)) {
        throw new Error(`Unknown pointwise operation type: ${opType}`);
    }
    return pointwiseOpMap[opType];
} 