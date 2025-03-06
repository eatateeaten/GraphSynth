import torch
import abc
import uuid
from typing import Optional, Tuple, List, Dict, Union, Iterable, Iterator, Any
import tensorflow as tf 
import numpy as np 
from abc import ABC

Sample_String = " "

class Shape:
    """
    Class representing the shape of a tensor.
    """
    def __init__(self, dimensions: Iterable[int]):
        """
        Initialize a Shape object with the given dimensions.
        
        Parameters:
        - dimensions: An iterable of integers representing the dimensions
        """
        self.dimensions = tuple(int(dim) for dim in dimensions)
        self._validate()
    
    def _validate(self):
        """
        Validate that all dimensions are valid.
        """
        for i, dim in enumerate(self.dimensions):
            if dim <= 0:
                raise InvalidShapeError(f"Invalid dimension at index {i}: {dim}. All dimensions must be positive.")
    
    def __eq__(self, other) -> bool:
        """
        Check if this shape is equal to another shape.
        """
        if isinstance(other, Shape):
            return self.dimensions == other.dimensions
        elif isinstance(other, tuple):
            return self.dimensions == other
        return False
    
    def __ne__(self, other) -> bool:
        """
        Check if this shape is not equal to another shape.
        """
        return not self.__eq__(other)
    
    def __len__(self) -> int:
        """
        Get the number of dimensions.
        """
        return len(self.dimensions)
    
    def __getitem__(self, index) -> int:
        """
        Get a dimension by index.
        """
        return self.dimensions[index]
    
    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the dimensions.
        """
        return iter(self.dimensions)
    
    def __repr__(self) -> str:
        """
        Get a string representation of the shape.
        """
        return f"Shape{self.dimensions}"
    
    def __str__(self) -> str:
        """
        Get a string representation of the shape.
        """
        return str(self.dimensions)
    
    def total_elements(self) -> int:
        """
        Calculate the total number of elements in a tensor with this shape.
        """
        if not self.dimensions:
            return 0
        total = 1
        for dim in self.dimensions:
            total *= dim
        return total


class Seq(): 
    def __init__(self, sequence: Optional[List[Node]] = None, shape_match_check = True):
        """
        Parameters:
        - sequence: A list of Node objects.
        - check_dimension: A flag to check dimensions.
        """
        self.sequence = sequence if sequence is not None else []
        if shape_match_check:
            
        return 

    def check_shape_match(): 
        pass 
    
    def __str__(): 
        """return the serialized string of a sequence"""
        
        return 
    
    def connect_from():
        return 
     
    
    def connect_to():
        return 


class Graph(): 
    def __init__(self, seqs: Optional[List[Seq]] = None, edges=None, shape_match_check=True):
        """
        Initialize a graph using an adjacency list representation.
        
        Parameters:
        - seqs: Initial list of Seq objects to add to the graph, or None
        - edges: Initial list of edges as tuples (from_seq, operation_node, to_seq)
        - shape_match_check: If True, validates shape compatibility when adding edges
        """
        # Dictionary mapping sequences to list of (operation, destination) tuples
        self.seqs: Dict[Seq, List[Tuple[Node, Seq]]] = {}
        self.shape_match_check = shape_match_check 
        
        # Initialize with provided sequences
        if seqs:
            for seq in seqs:
                self.add_seq(seq)
        
        # Initialize with provided edges
        if edges:
            for from_seq, operation, to_seq in edges:
                self.add_edge(from_seq, operation, to_seq)
    
    def add_seq(self, seq: 'Seq'):
        """
        Add a sequence to the graph.
        
        Parameters:
        - seq: The sequence to add to the graph
        """
        if seq not in self.seqs:
            self.seqs[seq] = []
    
    def add_edge(self, from_seq: 'Seq', operation: Union['MergeNode', 'BranchNode'], to_seq: 'Seq'):
        """
        Add a directed edge from one sequence to another through an operation node.
        
        Parameters:
        - from_seq: Source sequence
        - operation: The operation node that connects from_seq to to_seq (e.g., merge or branch)
        - to_seq: Destination sequence
        """
        # Add sequences if they don't exist
        if from_seq not in self.seqs:
            self.add_seq(from_seq)
        if to_seq not in self.seqs:
            self.add_seq(to_seq)
            
        # Check shape compatibility if required
        if self.shape_match_check:
            # Verify from_seq's output shape matches operation's input shape
            if not Node.shapes_equal(from_seq.out_shape, operation.in_shape):
                raise InShapeMismatchError(
                    f"Cannot connect {from_seq} to {operation}: output shape {from_seq.out_shape} "
                    f"doesn't match operation input shape {operation.in_shape}"
                )
            
            # Verify operation's output shape matches to_seq's input shape
            if not Node.shapes_equal(operation.out_shape, to_seq.in_shape):
                raise OutShapeMismatchError(
                    f"Cannot connect {operation} to {to_seq}: operation output shape {operation.out_shape} "
                    f"doesn't match destination input shape {to_seq.in_shape}"
                )
        
        # Add edge to adjacency list
        self.seqs[from_seq].append((operation, to_seq))
        
        # Update the connections
        from_seq.out_node = operation
        operation.in_node = from_seq
        operation.out_node = to_seq
        to_seq.in_node = operation
    
    def get_successors(self, seq: 'Seq'):
        """
        Get all sequences that the given sequence connects to.
        
        Parameters:
        - seq: The sequence to get successors for
        
        Returns:
        - List of (operation, destination) tuples
        """
        return self.seqs.get(seq, [])
    
    def get_destination_seqs(self, seq: 'Seq'):
        """
        Get only the destination sequences that the given sequence connects to.
        
        Parameters:
        - seq: The sequence to get destination sequences for
        
        Returns:
        - List of destination sequences
        """
        return [dest_seq for _, dest_seq in self.seqs.get(seq, [])]
    
    def get_predecessors(self, seq: 'Seq'):
        """
        Get all sequences that connect to the given sequence.
        
        Parameters:
        - seq: The sequence to get predecessors for
        
        Returns:
        - List of (source_seq, operation) tuples
        """
        predecessors = []
        for source_seq, edges in self.seqs.items():
            for operation, dest_seq in edges:
                if dest_seq == seq:
                    predecessors.append((source_seq, operation))
        return predecessors
    
    def get_edges(self):
        """
        Get all edges in the graph.
        
        Returns:
        - List of tuples representing edges (from_seq, operation, to_seq)
        """
        edges = []
        for from_seq, connections in self.seqs.items():
            for operation, to_seq in connections:
                edges.append((from_seq, operation, to_seq))
        return edges
    
    def topological_sort(self):
        """
        Perform a topological sort of the graph.
        
        Returns:
        - List of sequences and operations in topological order
        """
        visited = set()
        temp = set()
        order = []
        
        # Create a combined graph of sequences and operations
        combined_graph = {}
        for from_seq, connections in self.seqs.items():
            if from_seq not in combined_graph:
                combined_graph[from_seq] = []
            
            for operation, to_seq in connections:
                combined_graph[from_seq].append(operation)
                if operation not in combined_graph:
                    combined_graph[operation] = []
                combined_graph[operation].append(to_seq)
                if to_seq not in combined_graph:
                    combined_graph[to_seq] = []
        
        def visit(node):
            if node in temp:
                raise ValueError("Graph has a cycle")
            if node in visited:
                return
            
            temp.add(node)
            for successor in combined_graph.get(node, []):
                visit(successor)
            
            temp.remove(node)
            visited.add(node)
            order.append(node)
        
        for node in combined_graph:
            if node not in visited:
                visit(node)
        
        return order[::-1]  # Reverse to get correct order
    
    def to_torch(self):
        """
        Convert the graph to PyTorch code.
        
        Returns:
        - String containing PyTorch code to recreate the graph
        """
        # Get elements in execution order
        sorted_elements = self.topological_sort()
        
        # Track variable names for each element
        var_names = {}
        code_lines = []
        
        for i, element in enumerate(sorted_elements):
            var_name = f"x{i}"
            var_names[element] = var_name
            
            if isinstance(element, Seq):
                if not any(element == src for src, _, _ in self.get_edges()):
                    # This is an input sequence
                    code_lines.append(f"{var_name} = input_tensor  # Input to {element}")
                else:
                    # This is a sequence that has inputs
                    predecessors = self.get_predecessors(element)
                    if predecessors:
                        # Use the output from the last operation that feeds into this sequence
                        prev_op, _ = predecessors[0]
                        prev_var = var_names[prev_op]
                        code_lines.append(f"{var_name} = {element.to_torch()}({prev_var})")
            else:
                # This is an operation node
                # Find its inputs
                inputs = []
                for src_seq, op, dst_seq in self.get_edges():
                    if op == element:
                        inputs.append(var_names[src_seq])
                
                if inputs:
                    inputs_str = ", ".join(inputs)
                    code_lines.append(f"{var_name} = {element.to_torch()}({inputs_str})")
        
        return "\n".join(code_lines)


class Node(abc.ABC):
    """
    Abstract Node class 
    An abstraction for all layers and aggregated NN-graph 
    """
    def __init__(
        self,
        type: str = "",
        params: dict = {},
        in_shape: Optional[Shape] = None,
        out_shape: Optional[Shape] = None,
        in_node: Optional["Node"] = None,
        out_node: Optional["Node"] = None,
    ):
        self.id = str(uuid.uuid4())  # Unique ID
        self.type= type or f"Node_{self.id}"
        self.params = params
        self.in_shape = in_shape
        self.out_shape = out_shape

        if self.out_shape is None and self.in_shape is not None:
            try:
                self.out_shape = self.forward_dimension_inference(self.in_shape)
            except (NotImplementedError, ValueError) as e:
                raise ValueError(f"Failed to infer out_shape: {str(e)}") 
            
        self.in_node: Optional["Node"] = in_node
        self.out_node: Optional["Node"] = out_node
    
    
    def __str__(self) -> str:
        """
        Return a readable summary of the node.
        """
        params_str = ", ".join(f"{param}={val}" for param, val in self.params.items())
        return f"{self.type}({params_str})"
    
    def completed(self):
        return self.in_shape is not None and self.out_shape is not None and self.in_node is not None and self.out_node is not None 
    
    
    @staticmethod
    def shapes_equal(shape1: Optional[Shape], shape2: Optional[Shape]) -> bool:
        """
        Check if two shapes are equal.

        Parameters:
        - shape1: The first shape, or None.
        - shape2: The second shape, or None.

        Returns:
        - True if both shapes are equal and not None, False otherwise.
        """
        if shape1 is None or shape2 is None:
            return False
        return shape1 == shape2

    @staticmethod
    def validate_shape(self, shape: Shape) -> str:
        """
        check if a shape is a valid tensor shape
        check shape against overflow, underflow, dividing-by-zero, etc. 
        """
        for i in range(len(shape)): 
            if shape[i] <= 0:  
                raise InvalidShapeError(f"Invalid shape {shape}: {i}-th dim {shape[i]} must be larger than 0")
        return 
    

    @staticmethod
    def recompute_out_shape_from_in_shape(self): 
        if not self.in_shape: 
            self.out_shape 
            return 
        self.out_shape = this. 


    
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
    
    

    def set_input_node(self, other_node: 'Node'):
        """
        Link a node to the input of this node, throw error if the other_node's out_shape is defined and mismatched with this node's in_shape
        """
        if self.in_shape is None:
            self.set_in_shape(other_node.out_shape)
        if self.in_shape != other_node.out_shape:
            raise InShapeMismatchError(f"Invalid add node, link node {self} to the output of {other_node}, {self} in_shape mismatch with {other_node} out_shape")
        self.in_node = other_node
        other_node.out_node = self 

    def set_output_node(self, other_node: 'Node'):
        """
        Link a node to the output of this node, throw error if the other_node's in_shape is defined and mismatched with this node's out_shape
        """
        if self.out_shape is None:
            self.set_out_shape(other_node.in_shape)
        if self.out_shape != other_node.in_shape:
            raise OutShapeMismatchError(f"Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        self.out_node = other_node 
        other_node.in_node = self


    @abc.abstractmethod
    def validate_params(self):
        """
        validate the layer parameters with requirements specific to each layer type 
        """
        pass

    @abc.abstractmethod
    def compute_out_shape_from_in_shape(self, in_shape) -> Shape:
        """
        infer out_shape from in_shape with layer type and layer parameters
        TODO  in some cases, the layer parameters already defined the out_shape, like Reshape 
        in these cases, we check if the defined operations can be applied on this in_shape, and throw errors if it cannot be,
        if it can be, it returns the predefined out_shape 
        """
        pass

    @abc.abstractmethod
    def to_torch(self) -> str:
        """
        Return a string with the PyTorch layer construction (e.g. 'nn.Linear(128, 64)').
        """
        pass




class Merge(ABC):
    """
    Abstract base class for merge operations. A Merge operation is any neurel network operation that takes multiple tensors and output one tensor
    """
    def __init__():
        return 
    
    def connect_from():
        return 

    def connect_to():
        return 
    
    def connect_to():
        return 
    
        
class Branch(ABC):
    """
    Abstract base class for branch operations. A Branch operation is any neurel network operation that takes one tensor and outputs multiple tensors 
    """
    def __init__():
        return 
    
    def connect_from():
        return 

    
    def connect_to():
        return 
    
    def connect_to():
        return 

# Example subclass
class ConcatenateMerge(Merge):
    def perform(self, *args, **kwargs):
        print("Performing concatenation merge")

class AddMerge(Merge):
    def perform(self, *args, **kwargs):
        print("Performing addition merge")




class Tensor(Node):
    ## Tensor Type should be drawn with a different UI than other nodes 
    ## You should not allow people edit its input ouput shape
    """
    Represents a tensor node
    It's usually defined as an input and output node of a Seq 
    """
    def __init__(self, data: torch.Tensor| tf.Tensor | np.ndarray):
        super().__init__(type="Tensor", in_shape=Shape(data.shape) , out_shape=Shape(data.shape)) # The in_shape and out_shape of a Tensor type equals to its data shape. It is immutable and will never be None once the Node is defined. 
        self.data = data

    def forward_dimension_inference(self, in_shape: Shape) -> Shape:
        """
        For a tensor, the output shape is the same as the input shape.
        """ 
        return in_shape 

    def to_torch(self) -> str:
        """
        Return a string with the PyTorch tensor construction.
        Convert self.data to torch.Tensor if it is not already.
        """
        if isinstance(self.data, torch.Tensor):
            return f"torch.tensor({self.data.tolist()})"
        elif isinstance(self.data, tf.Tensor):
            data = torch.tensor(self.data.numpy())
        elif isinstance(self.data, np.ndarray):
            data = torch.tensor(self.data)
        else:
            raise TypeError("Unsupported data type for conversion to torch.Tensor")
        return f"torch.tensor({data.tolist()})"

    def to_tf(self) -> str:
        """
        Return a string with the TensorFlow tensor construction.
        Convert self.data to tf.Tensor if it is not already.
        """
        if isinstance(self.data, tf.Tensor):
            return f"tf.convert_to_tensor({self.data.numpy().tolist()})"
        elif isinstance(self.data, torch.Tensor):
            data = tf.convert_to_tensor(self.data.numpy())
        elif isinstance(self.data, np.ndarray):
            data = tf.convert_to_tensor(self.data)
        else:
            raise TypeError("Unsupported data type for conversion to tf.Tensor")
        return f"tf.convert_to_tensor({data.numpy().tolist()})"

    def validate_params(self):
        """
        Validate that the tensor data is consistent with the shape.
        """
        if self.data.shape != self.in_shape:
            raise ValueError("Tensor data shape does not match the specified in_shape")

    def set_in_shape(self, new_in_shape) -> Node:
        """Overrides the Node Class method"""
        raise ImmutableInShapeError("Invalid Call. Cannot change in_shape of a Tensor. In_shape is immutable and must match the data shape. Consider creating a new Tensor")

    def set_out_shape(self, new_out_shape) -> Node:
        """Overrides the Node Class method"""
        raise ImmutableOutShapeError("Invalid Call. Cannot change out_shape of a Tensor. Out_shape is immutable and must match the data shape. Consider creating a new Tensor")

    def set_input_node(self, other_node: 'Node'):
        """
        Overrides the Node Class method
        Link a node to the input of this node, throw error if the other_node's out_shape is defined and mismatched with this node's in_shape
        """
        if self.in_shape is None:
            raise ImmutableInShapeError("Tensor in_shape should never be None.")
        if self.in_shape != other_node.out_shape:
            raise InShapeMismatchError(f"Invalid add node, link node {self} to the output of {other_node}, {self} in_shape mismatch with {other_node} out_shape")
        self.in_node = other_node
        other_node.out_node = self

    def set_output_node(self, other_node: 'Node'):
        """
        Overrides the Node Class method
        Link a node to the output of this node, throw error if the other_node's in_shape is defined and mismatched with this node's out_shape
        """
        if self.out_shape is None:
            raise ImmutableOutShapeError("Tensor out_shape should never be None.")
        if self.out_shape != other_node.in_shape:
            raise OutShapeMismatchError(f"Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        self.out_node = other_node
        other_node.in_node = self


class Reshape(Node):
    ## I'm working to make Reshape and Swap_dim automatic, but for now make Reshape a specific node ig
    """
    Maps to reshape in Torch 
    """
    def __init__(self, out_dim):
        super().__init__(type="Reshape", params= {"out_dim": out_dim})

    def forward_dimension_inference(self, in_shape) -> Shape:
        out_dim = self.params['out_dim']
        if not isinstance(out_dim, tuple):
            raise ForwardDimensionInferenceFailureError("out_dim must be a tuple")
        in_elements = torch.prod(torch.tensor(in_shape)).item()
        out_elements = 1
        inferred_index = -1
        for i, dim in enumerate(out_dim):
            if dim == -1:
                if inferred_index != -1:
                    raise ForwardDimensionInferenceFailureError("Only one dimension can be inferred, redefine the reshape parameters to contain only one -1")
                inferred_index = i
            else:
                out_elements *= dim
        if inferred_index != -1:
            if in_elements % out_elements != 0:
                raise ForwardDimensionInferenceFailureError("Cannot infer dimension: total elements do not match")
            inferred_dim = in_elements // out_elements
            out_dim = out_dim[:inferred_index] + (inferred_dim,) + out_dim[inferred_index+1:]
        elif in_elements != out_elements:
            raise ForwardDimensionInferenceFailureError(f"Cannot reshape from {in_shape} to {out_dim}: total elements do not match")
        return Shape(out_dim)

    def validate_params(self):
        out_dim = self.params.get('out_dim')
        if not isinstance(out_dim, tuple):
            raise InvalidLayerParametersError("out_dim must be a tuple")

        inferred_count = 0
        for dim in out_dim:
            if dim == -1:
                inferred_count += 1
                if inferred_count > 1:
                    raise InvalidLayerParametersError("Only one dimension can be inferred (-1) in out_dim")
            elif dim <= 0:
                raise InvalidLayerParametersError(f"Invalid dimension {dim} in out_dim: must be positive or -1")

    def to_torch(self) -> str:
        out_dim = self.params['out_dim']
        return f"torch.reshape(input, {out_dim})"


class NodeError(Exception, ABC):
    ## Abstract don't need to be handled 
    """
    Abstract base class for errors related to Node operations.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class InvalidShapeError(NodeError):
    ## Can appear in the chatbox on the right side
    """
    Exception raised for errors in the shape of a tensor or node.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message: str = ""):
        default_message = "Invalid shape provided."
        super().__init__(f"{default_message} {message}")


class InvalidLayerParametersError(NodeError):
    ## Can appear in the chatbox on the right side 
    """
    Exception raised for errors in the parameters of a layer.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message: str = ""):
        default_message = "Invalid layer parameters."
        super().__init__(f"{default_message} {message}")


class NoInputNodeError(NodeError):
    ## This one will be rare and is not an intended feature if all things work out 
    ## You don't need to implement anything for it on the front-end yet 
    def __init__(self, message: str = ""):
        default_message = "No input node connected."
        super().__init__(f"{default_message} {message}")


class NoOutputNodeError(NodeError):
    ## This one will be rare and is not an intended feature if all things work out 
    ## You don't need to implement anything for it on the front-end yet 
    def __init__(self, message: str = ""): 
        default_message = "No output node connected."
        super().__init__(f"{default_message} {message}")


class ForwardDimensionInferenceFailureError(NodeError):
    ## The bubble can become a pinkish red? 
    def __init__(self, message: str = ""):
        default_message = "Failed to infer forward dimensions."
        super().__init__(f"{default_message} {message}")


class ImmutableInShapeError(NodeError):
    ## Just don't give people the option to change a Tensor shape, this will never occur
    def __init__(self, message: str = ""):
        default_message = "Attempt to set immutable input shape."
        super().__init__(f"{default_message} {message}")


class ImmutableOutShapeError(NodeError):
    ## Just don't give people the option to change a Tensor shape, this will never occur 
    def __init__(self, message: str = ""):
        default_message = "Attempt to set immutable output shape."
        super().__init__(f"{default_message} {message}")


class InShapeMismatchError(NodeError):
    ## the left side of the bubble become red. the left side can just be the left edge. You can put a very thin rectangle there on the left edge of the bubble. And it becomes red if it bugs. When it works the in_shape can be displayed when you hover over the rectangle  
    def __init__(self, message: str = ""):
        default_message = "Input shape mismatch with previous output shape."
        super().__init__(f"{default_message} {message}")


class OutShapeMismatchError(NodeError):
    ## the right side of the bubble become red. the left side can just be the left edge. You can put a very thin rectangle there on the left edge of the bubble. And it becomes red if it bugs. When it works the in_shape can be displayed when you hover over the rectangle  
    def __init__(self, message: str = ""):
        default_message = "Output shape mismatch with subsequent input shape."
        super().__init__(f"{default_message} {message}")
