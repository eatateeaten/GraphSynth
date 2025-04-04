/**
 * Simple assertion function that throws an error if the condition is falsy
 */
export function assert(condition: any, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}
