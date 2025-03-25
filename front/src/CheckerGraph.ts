/**
 * TypeScript port of the Python shape-checking graph.
 * 
 * Key implementation notes:
 * - Shape = number[] (equivalent to Python List[int])
 * - Uses TypeScript classes instead of Python ABC
 * - Each node must implement:
 *   1. compute_out_shape(in_shape: Shape): Shape
 *   2. validate_params(): void
 * 
 * The main difference is how output shapes work:
 * - out_shape is only recomputed when inputs or params change
 * - we raise an error if the shape doesn't match the next node's input shape
 *      UI is expected to disconnect the nodes in this case
 * - compute_out_shape() is pure - just takes in_shape & params and returns result
 * - recompute_out_shape() handles the caching and error handling
 * - so in that sense I assumed the left to right building thingy
 */


export type Shape = number[]; //defining shape in primitive type 

function validate_shape(shape: Shape): void {
  if (shape.some(dim => dim <= 0)) {  //checking against smaller than 0 shape 
    const badDim = shape.findIndex(dim => dim <= 0); 
    throw new Error(`Invalid shape ${shape}: ${badDim}-th dim ${shape[badDim]} must be larger than 0`);
  }
}

/**checking if two shapes are equal 
 * return True if they are both not None, same in length, and each component are the same 
 * else return False 
 */
function shapes_equal(shape1: Shape | null, shape2: Shape | null): boolean { 
  if (shape1 === null || shape2 === null) return false;
  if (shape1.length !== shape2.length) return false;   
  return shape1.every((dim, i) => dim === shape2[i]);
}

/**
 * 
 */
export abstract class CheckerNode {
  in_shape: Shape | null = null;
  out_shape: Shape | null = null;
  in_node: CheckerNode | null = null;
  out_node: CheckerNode | null = null;
  
  //TODO add params, type to the field
  //Handle printing the node 
  
  constructor(protected params: Record<string, any> = {}, in_shape: Shape | null) {
    this.validate_params();
    this.in_shape = in_shape; 
    if (this.in_shape !== null) {
      this.recompute_out_shape();
    }  
  } 

  abstract compute_out_shape(in_shape: Shape): Shape;
  abstract validate_params(): void;

  protected recompute_out_shape(): void {
    if (this.in_shape === null) {
      this.out_shape = null;
      return;
    } 
    this.out_shape = this.compute_out_shape(this.in_shape); 
    // If shape changes and this node has an out node and it doesn't match the next node's input shape, throw an error
    if (this.out_node && !shapes_equal(this.out_shape, this.out_node.in_shape)) {
      throw new Error(`Output shape mismatch with subsequent input shape: ${this.out_shape} vs ${this.out_node.in_shape}`);
    }
  }

  set_params(params: Record<string, any>): void {
    this.params = params;
    this.validate_params();
    this.recompute_out_shape();
  }

  connect_to(target: CheckerNode): void {
    if (this.in_shape !== null) {
      if (target.in_shape !== null) {
        if (!shapes_equal(this.out_shape, target.in_shape)) {
          throw new Error(`Input shape mismatch: cannot connect output shape ${this.out_shape} to input shape ${target.in_shape}`);
        }
      } else {
        target.set_in_shape(this.out_shape);
      }
    }
    this.out_node = target;
    target.in_node = this;
  }

  set_in_shape(shape: Shape | null): void {
    if (shape !== null) {
      validate_shape(shape);
    }
    this.in_shape = shape;
    this.recompute_out_shape();
  }
}

export class Tensor extends CheckerNode {
  constructor(params: Record<string, any>, in_shape: Shape | null) {
    super(params, in_shape);
    validate_shape(params.shape);
    this.in_shape = params.shape;  // Tensors have fixed shape
    this.recompute_out_shape();
  }

  validate_params(): void {
    const { shape } = this.params;
    if (!Array.isArray(shape)) {
      throw new Error("shape must be an array");
    }
    validate_shape(shape);
  }

  compute_out_shape(in_shape: Shape): Shape {
    return in_shape;  // Tensor just passes through its shape
  }

  set_in_shape(shape: Shape | null): void {
    if (shape !== null && !shapes_equal(shape, this.in_shape)) {
      throw new Error("Cannot change tensor shape: tensor shapes are immutable");
    }
  }
}

export class Reshape extends CheckerNode {
  constructor(params: Record<string, any>) {
    super(params);
  }

  validate_params(): void {
    const { out_dim } = this.params;
    if (!Array.isArray(out_dim)) {
      throw new Error("out_dim must be an array");
    }

    const inferredDims = out_dim.filter(d => d === -1);
    if (inferredDims.length > 1) {
      throw new Error("Only one dimension can be inferred (-1) in out_dim");
    }

    const invalidDims = out_dim.filter(d => d !== -1 && d <= 0);
    if (invalidDims.length > 0) {
      throw new Error(`Invalid dimension ${invalidDims[0]} in out_dim: must be positive or -1`);
    }
  }

  compute_out_shape(in_shape: Shape): Shape {
    const out_dim = this.params.out_dim as number[];
    const total_in = in_shape.reduce((a, b) => a * b, 1);
    
    if (out_dim.includes(-1)) {
      const known_out = out_dim.filter(d => d !== -1).reduce((a, b) => a * b, 1);
      const missing_dim = total_in / known_out;
      if (!Number.isInteger(missing_dim)) {
        throw new Error("Cannot compute missing dimension: total elements do not match");
      }
      return out_dim.map(d => d === -1 ? missing_dim : d);
    }

    const total_out = out_dim.reduce((a, b) => a * b, 1);
    if (total_in !== total_out) {
      throw new Error(`Cannot reshape from ${in_shape} to ${out_dim}: total elements do not match`);
    }
    return out_dim;
  }
}

export class CheckerGraph {
  private nodes = new Map<string, CheckerNode>();

  addNode(id: string, node: CheckerNode): void {
    this.nodes.set(id, node);
  }

  getNode(id: string): CheckerNode | undefined {
    return this.nodes.get(id);
  }

  connect(sourceId: string, targetId: string): void {
    const source = this.nodes.get(sourceId);
    const target = this.nodes.get(targetId);
    
    if (!source || !target) {
      throw new Error("Node not found");
    }

    source.connect_to(target);
  }

  deleteNode(id: string): void {
    const node = this.nodes.get(id);
    if (!node) return;

    if (node.in_node) {
      node.in_node.out_node = null;
    }
    if (node.out_node) {
      node.out_node.in_node = null;
    }

    this.nodes.delete(id);
  }
}
