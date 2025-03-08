export type Shape = number[];

export const Shape = {
    equals(a: Shape | null, b: Shape | null): boolean {
        if (a === null || b === null) return a === b;
        if (a.length !== b.length) return false;
        return a.every((dim, i) => dim === b[i]);
    },

    // Common shape validation functions
    validatePositive(shape: Shape, name: string = 'shape'): void {
        const badDim = shape.findIndex(dim => dim <= 0);
        if (badDim !== -1) {
            throw new Error(`Invalid ${name} ${shape}: ${badDim}-th dim ${shape[badDim]} must be positive`);
        }
    },

    validateSpatialDims(shape: Shape, expectedDims: number, name: string = 'shape'): void {
        if (shape.length !== expectedDims + 2) {
            throw new Error(
                `Invalid ${name} ${shape}: expected ${expectedDims + 2} dimensions ` +
                `(batch, channels, spatial...), got ${shape.length}`
            );
        }
    },

    validateChannels(shape: Shape, expectedChannels: number): void {
        if (shape[1] !== expectedChannels) {
            throw new Error(
                `Channel mismatch: expected ${expectedChannels}, got ${shape[1]}`
            );
        }
    },

    // For reshape operations
    validateReshapeShape(shape: Shape): void {
        const inferredDims = shape.filter(d => d === -1);
        if (inferredDims.length > 1) {
            throw new Error('Only one dimension can be inferred (-1)');
        }
        
        const invalidDims = shape.filter(d => d !== -1 && d <= 0);
        if (invalidDims.length > 0) {
            throw new Error(`Invalid dimension ${invalidDims[0]}: must be positive or -1`);
        }
    }
} as const;
