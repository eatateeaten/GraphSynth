export class Shape extends Array<number> {
    constructor(dimensions: number[], allowNegativeOne = false) {
        super(...dimensions);
        this.validate(allowNegativeOne);
    }

    static fromArray(dims: number[], allowNegativeOne = false): Shape {
        return new Shape(dims, allowNegativeOne);
    }

    static fromNumber(dim: number): Shape {
        return new Shape([dim]);
    }

    static fromString(str: string, allowNegativeOne = false): Shape {
        const dims = str.split(',')
            .map(s => parseInt(s.trim(), 10))
            .filter(n => !isNaN(n));
        
        if (dims.length === 0) {
            throw new Error('Invalid shape string: no valid numbers found');
        }
        return new Shape(dims, allowNegativeOne);
    }

    private validate(allowNegativeOne: boolean): void {
        const badDim = this.findIndex(dim => 
            allowNegativeOne 
                ? (dim !== -1 && dim <= 0)
                : dim <= 0
        );
        if (badDim !== -1) {
            const msg = allowNegativeOne 
                ? 'must be positive or -1'
                : 'must be positive';
            throw new Error(`Invalid shape ${this}: ${badDim}-th dim ${this[badDim]} ${msg}`);
        }
    }

    toString(): string {
        return `[${this.join(', ')}]`;
    }

    equals(other: Shape | null): boolean {
        if (other === null) return false;
        if (this.length !== other.length) return false;
        return this.every((dim, i) => dim === other[i]);
    }
}
