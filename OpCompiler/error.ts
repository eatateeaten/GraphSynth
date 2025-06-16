/** Custom error types */
const createError = (name: string) => class extends Error {
    constructor(message: string) {
        super(message);
        this.name = name;
    }
};

export const ShapeMatchError = createError("ShapeMatchError");
export const ShapeInferenceError = createError("ShapeInferenceError"); 
export const ParamError = createError("ParamError"); 
export const TargetError = createError("TargetError");
export const CycleError = createError("CycleError");
export const SourceNotTensorError = createError("SourceNotTensorError");
export const SinkNotTensorError = createError("SinkNotTensorError");
export const UnreachableNodeError = createError("UnreachableNodeError"); 