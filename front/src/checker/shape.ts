export type Shape = number[];

export function validate_shape(shape: Shape): void {
    const badDim = shape.findIndex(dim => dim <= 0);
    if (badDim !== -1) {
        throw new Error(`Invalid shape ${shape}: ${badDim}-th dim ${shape[badDim]} must be larger than 0`);
    }
}

export function shapes_equal(shape1: Shape | null, shape2: Shape | null): boolean {
    if (shape1 === null || shape2 === null) return false;
    if (shape1.length !== shape2.length) return false;
    return shape1.every((dim, i) => dim === shape2[i]);
}
