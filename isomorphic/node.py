import torch
import uuid
from typing import Optional, Tuple, List, Dict, Union 
import tensorflow as tf 
import numpy as np 
from abc import ABC
from .shape import Shape, InvalidShapeError
from .torch_nn_module_calls import validate_params, get_torch_code

##
##Node - NN-ops 
##
class Node:
    """
    An abstraction for all sequential NN-op and aggregated NN-ops 
    """
    def __init__( 
        self,
        type: str, 
        params: dict = {},
        in_shape: Optional[Shape] = None, 
        out_shape: Optional[Shape] = None, 
        in_node: Optional["Node"] = None,  
        out_node: Optional["Node"] = None,  
    ):
        self.type = type
        self.params = params
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_node: Optional["Node"] = in_node
        self.out_node: Optional["Node"] = out_node 
        
        # Validate parameters if this type is in our metadata dictionary
        validate_params(self.type, self.params)
    
    def __str__(self) -> str:
        """
        Return a readable summary of the node.
        """
        params_str = ", ".join(f"{param}={val}" for param, val in self.params.items())
        return f"{self.type}({params_str})"
    

    def to_torch(self) -> str:
        """
        Return a string with the PyTorch layer construction.
        Uses get_torch_code function for validation and code generation.
        """
        return get_torch_code(self.type, self.params)

    def to_jax(self) -> str:
        """
        Not implemented. TODO We will deal with this later 
        """
        pass 


class Merge(Node):
    """
    A class for merging NN-op that takes multiple tensors and gives one tensor.
    """
    def __init__(self, input_nodes: List[Node], output_node: Optional[Node] = None, params: dict = None, **kwargs):
        params = params or {}
        super().__init__(type="Merge", params=params, **kwargs)
        self.input_nodes = input_nodes or []
        self.output_node = output_node


class Branch(Node):
    """
    A class for branching NN-op that takes one tensor and gives multiple tensors.
    """
    def __init__(self, input_node: Node, output_nodes: Optional[List[Node]] = None, params: dict = None, **kwargs):
        params = params or {}
        super().__init__(type="Branch", params=params, **kwargs)
        self.in_node = input_node
        self.output_nodes = output_nodes or []


class ConcatMerge(Merge):
    """
    Concatenates multiple tensors along a specified dimension.
    """
    def __init__(self, input_nodes: List[Node], params: dict = None, output_node: Optional[Node] = None, **kwargs):
        params = params or {}
        concat_dim = params.get('concat_dim', 0)
        merge_params = {'concat_dim': concat_dim}
        super().__init__(input_nodes=input_nodes, output_node=output_node, params=merge_params, **kwargs)
        self.type = "ConcatMerge"
        
    def validate_shapes(self):
        """Validate that input tensors have compatible shapes for concatenation."""
        if not self.input_nodes or len(self.input_nodes) < 2:
            raise ValueError("ConcatMerge requires at least two input tensors")
        
        shapes = [node.out_shape for node in self.input_nodes]
        concat_dim = self.params['concat_dim']
        
        # Check that all shapes have the same dimensionality
        ndims = len(shapes[0])
        if any(len(shape) != ndims for shape in shapes):
            raise ValueError("All input tensors must have the same number of dimensions")
            
        # Check that all dimensions except concat_dim match
        for dim in range(ndims):
            if dim != concat_dim:
                if any(shape[dim] != shapes[0][dim] for shape in shapes):
                    raise ValueError(f"Input tensors must have the same size in all dimensions except concat_dim, mismatch found in dimension {dim}")
    
    def to_torch(self) -> str:
        """Return PyTorch code for tensor concatenation."""
        return f"torch.cat(tensors=[{', '.join(f'x{i}' for i, _ in enumerate(self.input_nodes))}], dim={self.params['concat_dim']})"


class PositionwiseMerge(Merge):
    """
    Applies element-wise operations (default: addition) to input tensors.
    """
    def __init__(self, input_nodes: List[Node], params: dict = None, output_node: Optional[Node] = None, **kwargs):
        params = params or {}
        op = params.get('op', 'add')
        merge_params = {'op': op}
        super().__init__(input_nodes=input_nodes, output_node=output_node, params=merge_params, **kwargs)
        self.type = "PositionwiseMerge"
        
    def validate_shapes(self):
        """Validate that input tensors have the same shape."""
        if not self.input_nodes or len(self.input_nodes) < 2:
            raise ValueError("PositionwiseMerge requires at least two input tensors")
            
        shapes = [node.out_shape for node in self.input_nodes]
        
        # Check that all shapes are identical
        for shape in shapes[1:]:
            if shape != shapes[0]:
                raise ValueError(f"All input tensors must have the same shape. Found {shapes[0]} and {shape}")
    
    def to_torch(self) -> str:
        """Return PyTorch code for element-wise operation."""
        if self.params['op'] == 'add':
            # For two inputs, use simple addition
            if len(self.input_nodes) == 2:
                return f"(x0 + x1)"
            # For more inputs, use sum
            else:
                return f"torch.sum(torch.stack([{', '.join(f'x{i}' for i, _ in enumerate(self.input_nodes))}]), dim=0)"
        elif self.params['op'] == 'multiply':
            # For two inputs, use simple multiplication
            if len(self.input_nodes) == 2:
                return f"(x0 * x1)"
            # For more inputs, use product
            else:
                return f"torch.prod(torch.stack([{', '.join(f'x{i}' for i, _ in enumerate(self.input_nodes))}]), dim=0)"
        else:
            # For custom operations, this would need to be handled accordingly
            return f"custom_op([{', '.join(f'x{i}' for i, _ in enumerate(self.input_nodes))}])"


class ContractionMerge(Merge):
    """
    Performs contraction operations (like matrix multiplication) between tensors.
    """
    def __init__(self, input_nodes: List[Node], params: dict = None, output_node: Optional[Node] = None, **kwargs):
        params = params or {}
        contraction_dim = params.get('contraction_dim', 0)
        op = params.get('op', 'multiply')
        agg_op = params.get('agg_op', 'add')
        merge_params = {
            'contraction_dim': contraction_dim,
            'op': op,
            'agg_op': agg_op
        }
        super().__init__(input_nodes=input_nodes, output_node=output_node, params=merge_params, **kwargs)
        self.type = "ContractionMerge"
    
    def validate_shapes(self):
        """Validate that input tensors have compatible shapes for contraction."""
        if not self.input_nodes or len(self.input_nodes) != 2:
            raise ValueError("ContractionMerge currently supports exactly two input tensors")
            
        shape1 = self.input_nodes[0].out_shape
        shape2 = self.input_nodes[1].out_shape
        
        # Check dimensions for matrix multiplication
        if len(shape1) != len(shape2):
            raise ValueError(f"Input tensors must have the same number of dimensions. Found {shape1} and {shape2}")
    
    def to_torch(self) -> str:
        """Return PyTorch code for tensor contraction."""
        return f"torch.matmul(x0, x1)"  # Simplified for common case


class OuterProductMerge(Merge):
    """
    Performs outer product operations between tensors.
    """
    def __init__(self, input_nodes: List[Node], params: dict = None, output_node: Optional[Node] = None, **kwargs):
        params = params or {}
        outer_product_dim = params.get('outer_product_dim', 0)
        op = params.get('op', 'multiply')
        merge_params = {
            'outer_product_dim': outer_product_dim,
            'op': op
        }
        super().__init__(input_nodes=input_nodes, output_node=output_node, params=merge_params, **kwargs)
        self.type = "OuterProductMerge"
    
    def validate_shapes(self):
        """Validate input tensor shapes for outer product."""
        if not self.input_nodes or len(self.input_nodes) != 2:
            raise ValueError("OuterProductMerge currently supports exactly two input tensors")
    
    def to_torch(self) -> str:
        """Return PyTorch code for outer product."""
        return f"torch.einsum('i,j->ij', x0, x1)"  # Simplified for common case


class AttentionMerge(Merge):
    """
    Applies attention mechanism to merge tensors.
    """
    def __init__(self, input_nodes: List[Node], params: dict = None, output_node: Optional[Node] = None, **kwargs):
        params = params or {}
        num_heads = params.get('num_heads', 1)
        embed_dim = params.get('embed_dim', None)
        merge_params = {
            'num_heads': num_heads
        }
        if embed_dim:
            merge_params['embed_dim'] = embed_dim
            
        # Expecting input_nodes to be in order: [query, key, value]
        if len(input_nodes) != 3:
            raise ValueError("AttentionMerge requires exactly three input tensors (query, key, value)")
            
        super().__init__(input_nodes=input_nodes, output_node=output_node, params=merge_params, **kwargs)
        self.type = "AttentionMerge"
    
    def validate_shapes(self):
        """Validate shapes for attention mechanism."""
        if len(self.input_nodes) != 3:
            raise ValueError("AttentionMerge requires exactly three input tensors (query, key, value)")
        
        # Check that all inputs have compatible shapes for attention
        query_shape = self.input_nodes[0].out_shape
        key_shape = self.input_nodes[1].out_shape
        value_shape = self.input_nodes[2].out_shape
        
        # Validate dimensions
        if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
            raise ValueError("Query, key, and value tensors must be 3D (batch_size, seq_length, embed_dim)")
        
        # Validate batch sizes match
        if query_shape[0] != key_shape[0] or query_shape[0] != value_shape[0]:
            raise ValueError(f"Batch sizes must match. Found query: {query_shape[0]}, key: {key_shape[0]}, value: {value_shape[0]}")
            
        # Validate embedding dimensions
        embed_dim = query_shape[2]
        if embed_dim != key_shape[2]:
            raise ValueError(f"Query and key must have same embedding dimension. Found query: {embed_dim}, key: {key_shape[2]}")
            
        # For multi-head attention, embedding dimension should be divisible by number of heads
        num_heads = self.params['num_heads']
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
            
        # Key and value should have the same sequence length
        if key_shape[1] != value_shape[1]:
            raise ValueError(f"Key and value must have same sequence length. Found key: {key_shape[1]}, value: {value_shape[1]}")
    
    def to_torch(self) -> str:
        """Return PyTorch code for attention mechanism."""
        embed_dim = self.input_nodes[0].out_shape[2]  # Get embedding dimension from query shape
        return f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={self.params['num_heads']})(query=x0, key=x1, value=x2)[0]"


class CopyBranch(Branch):
    """
    Copies a tensor to multiple downstream paths.
    """
    def __init__(self, input_node: Node, params: dict = None, output_nodes: Optional[List[Node]] = None, **kwargs):
        params = params or {}
        num_copies = params.get('num_copies', len(output_nodes) if output_nodes else 2)
        branch_params = {'num_copies': num_copies}
        
        if output_nodes is None:
            output_nodes = [None] * num_copies
            
        super().__init__(input_node=input_node, output_nodes=output_nodes, params=branch_params, **kwargs)
        self.type = "CopyBranch"
    
    def to_torch(self) -> str:
        """Return PyTorch code for tensor copying."""
        # Simple case: no actual operation needed, just reference the same tensor multiple times
        return f"x0"


class SplitBranch(Branch):
    """
    Splits a tensor along a specified dimension at specific indices.
    """
    def __init__(self, input_node: Node, params: dict = None, output_nodes: Optional[List[Node]] = None, **kwargs):
        """
        Initialize a SplitBranch.
        
        Args:
            input_node: The node providing the tensor to split
            params: Dictionary containing:
                - 'split_dim': The dimension along which to split the tensor
                - 'split_indices': List of indices at which to split (exclusive of 0 and tensor size)
            output_nodes: List of nodes to receive the split results (optional)
        """
        params = params or {}
        split_dim = params.get('split_dim', 0)
        split_indices = params.get('split_indices', [])
        
        # Create output nodes if not provided
        n_outputs = len(split_indices) + 1  # One more output than split points
        if output_nodes is None:
            output_nodes = [None] * n_outputs
        
        branch_params = {
            'split_dim': split_dim,
            'split_indices': split_indices
        }
        
        super().__init__(input_node=input_node, output_nodes=output_nodes, params=branch_params, **kwargs)
        self.type = "SplitBranch"
    
    def validate_shapes(self):
        """Validate that the split parameters are valid for the input tensor."""
        if not self.input_node or not self.input_node.out_shape:
            raise ValueError("Input node must have a defined output shape")
            
        input_shape = self.input_node.out_shape
        split_dim = self.params['split_dim']
        split_indices = self.params['split_indices']
        
        if split_dim >= len(input_shape):
            raise ValueError(f"Split dimension {split_dim} exceeds tensor dimensions {len(input_shape)}")
        
        dim_size = input_shape[split_dim]
        
        # Validate indices are in ascending order and within range
        if not all(0 < idx < dim_size for idx in split_indices):
            raise ValueError(f"Split indices must be between 0 and {dim_size} (exclusive)")
        if not all(split_indices[i] < split_indices[i+1] for i in range(len(split_indices)-1)):
            raise ValueError("Split indices must be in ascending order")
    
    def to_torch(self) -> str:
        """Return PyTorch code for tensor splitting."""
        split_dim = self.params['split_dim']
        split_indices = self.params['split_indices']
        
        return f"torch.split(x0, split_size_or_sections={split_indices}, dim={split_dim})"


class Seq(Node):
    """
    A sequence of nodes that can be treated as a single node or expanded into its components.
    Supports collapsible/expandable representation for UI and code generation.
    """
    def __init__(
        self, 
        nodes: List[Node], 
        name: str = None,  # Descriptive name for collapsed view (e.g., "ResNet Block")
        collapsed: bool = False,  # Default display state
        id: str = None,         #ID assigned from front-end 
        in_shape: Optional[Shape] = None, 
        out_shape: Optional[Shape] = None, 
        in_node: Optional["Node"] = None,  
        out_node: Optional["Node"] = None,
    ):
        # Generate a meaningful type name if none provided
        name = name or f"Seq_{str(uuid.uuid4())[:8]}"
        super().__init__(type=name, params={}, id=id, in_shape=in_shape, out_shape=out_shape, in_node=in_node, out_node=out_node)
        
        if not nodes:  # Catches None and empty lists
            raise ValueError("Seq must contain at least one Node")
        
        self.nodes = nodes
        self.collapsed = collapsed 
        
        # Connect internal nodes
        self._connect_nodes()
        
        # Calculate in_shape and out_shape based on first and last nodes
        self._update_shapes()
    
    def _connect_nodes(self):
        """Link all nodes in the sequence"""
        if not self.nodes:
            return
            
        for i in range(len(self.nodes) - 1):
            self.nodes[i].out_node = self.nodes[i + 1]
            self.nodes[i + 1].in_node = self.nodes[i]
    
    def _update_shapes(self):
        """Update sequence input and output shapes based on contained nodes"""
        if not self.nodes:
            return
            
        # Take shapes from first and last nodes
        if self.nodes[0].in_shape is not None:
            self.in_shape = self.nodes[0].in_shape
        
        if self.nodes[-1].out_shape is not None:
            self.out_shape = self.nodes[-1].out_shape 
    
    def collapse(self):
        """Collapse sequence to a single node representation"""
        self.collapsed = True
        return self
    
    def expand(self):
        """Expand sequence to show all component nodes"""
        self.collapsed = False
        return self
    
    def to_torch(self, expand: bool = None) -> str:
        """
        Generate PyTorch code, either as a Sequential or as a named module
        
        Args:
            expand: Override the sequence's collapsed state
        """
        # Determine if we should show expanded form
        is_expanded = self.collapsed is False if expand is None else expand
        
        if is_expanded:
            # Full expansion - generate Sequential with all nodes
            modules = [node.to_torch() for node in self.nodes]
            return f"nn.Sequential(\n    " + ",\n    ".join(modules) + "\n)"
        else:
            # Collapsed form - represent as a custom module
            return f"{self.type}()"  # Could include high-level params if needed
    
    def serialize(self) -> dict:
        """Serialize the sequence to a compact dict representation"""
        return {
            "type": self.type,
            "id": self.id,
            "collapsed": self.collapsed,
            "nodes": [self._serialize_node(node) for node in self.nodes],
            # Just store dimensions directly as a list (JSON-serializable)
            "in_shape": list(self.in_shape.dimensions) if self.in_shape else None,
            "out_shape": list(self.out_shape.dimensions) if self.out_shape else None,
        }
    
    @staticmethod
    def _serialize_node(node):          
        """Helper to serialize a node, handling both Node and Seq types"""
        if isinstance(node, Seq): 
            return node.serialize()
        # Simple serialization for basic nodes
        return {
            "id": node.id,
            "type": node.type,
            "params": node.params,
            # Direct dimension serialization
            "in_shape": list(node.in_shape.dimensions) if node.in_shape else None,
            "out_shape": list(node.out_shape.dimensions) if node.out_shape else None,
        }
    
    @classmethod
    def deserialize(cls, data: dict, node_registry=None) -> 'Seq':
        """
        Reconstruct a Seq from serialized data
        
        Args:
            data: Serialized sequence data
            node_registry: Optional dictionary mapping node IDs to existing Node instances
        """
        # Implementation depends on your node creation mechanism
        # This is a simplified example
        nodes = []
        for node_data in data["nodes"]:
            if "nodes" in node_data:  # This is a nested Seq
                nodes.append(Seq.deserialize(node_data, node_registry))
            else:
                # Create basic node
                node = Node(
                    type=node_data["type"],
                    params=node_data["params"]
                )
                nodes.append(node)
        
        # Direct Shape creation from dimensions list
        in_shape = Shape(data["in_shape"]) if data.get("in_shape") else None
        out_shape = Shape(data["out_shape"]) if data.get("out_shape") else None
        
        seq = cls(
            nodes=nodes,
            name=data["type"],
            collapsed=data.get("collapsed", False),
            in_shape=in_shape,
            out_shape=out_shape,
            id=data["id"]  # Pass ID directly in constructor
        )
        
        return seq


class Graph(): 
    def __init__(self, seqs: Optional[List[Seq]] = None, edges=None):
        """
        Initialize a graph using an adjacency list representation.
        
        Parameters:
        - seqs: Initial list of Seq objects to add to the graph, or None
        - edges: Initial list of edges as tuples (from_seq, operation_node, to_seq)
        - shape_match_check: If True, validates shape compatibility when adding edges
        """
        # Dictionary mapping sequences to list of (operation, destination) tuples
        self.seqs: Dict[Seq, List[Tuple[Node, Seq]]] = {}
        
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





class Seq(Node):
    """
    A sequence of nodes that can be treated as a single node or expanded into its components.
    Supports collapsible/expandable representation for UI and code generation.
    """
    def __init__(
        self, 
        nodes: List[Node], 
        name: str = None,  # Descriptive name for collapsed view (e.g., "ResNet Block")
        collapsed: bool = False,  # Default display state
        id: str = None,         #ID assigned from front-end 
        in_shape: Optional[Shape] = None, 
        out_shape: Optional[Shape] = None, 
        in_node: Optional["Node"] = None,  
        out_node: Optional["Node"] = None,
    ):
        # Generate a meaningful type name if none provided
        name = name or f"Seq_{str(uuid.uuid4())[:8]}"
        super().__init__(type=name, params={}, id=id, in_shape=in_shape, out_shape=out_shape, in_node=in_node, out_node=out_node)
        
        if not nodes:  # Catches None and empty lists
            raise ValueError("Seq must contain at least one Node")
        
        self.nodes = nodes
        self.collapsed = collapsed 
        
        # Connect internal nodes
        self._connect_nodes()
        
        # Calculate in_shape and out_shape based on first and last nodes
        self._update_shapes()
    
    def _connect_nodes(self):
        """Link all nodes in the sequence"""
        if not self.nodes:
            return
            
        for i in range(len(self.nodes) - 1):
            self.nodes[i].out_node = self.nodes[i + 1]
            self.nodes[i + 1].in_node = self.nodes[i]
    
    def _update_shapes(self):
        """Update sequence input and output shapes based on contained nodes"""
        if not self.nodes:
            return
            
        # Take shapes from first and last nodes
        if self.nodes[0].in_shape is not None:
            self.in_shape = self.nodes[0].in_shape
        
        if self.nodes[-1].out_shape is not None:
            self.out_shape = self.nodes[-1].out_shape 
    
    def collapse(self):
        """Collapse sequence to a single node representation"""
        self.collapsed = True
        return self
    
    def expand(self):
        """Expand sequence to show all component nodes"""
        self.collapsed = False
        return self
    
    def to_torch(self, expand: bool = None) -> str:
        """
        Generate PyTorch code, either as a Sequential or as a named module
        
        Args:
            expand: Override the sequence's collapsed state
        """
        # Determine if we should show expanded form
        is_expanded = self.collapsed is False if expand is None else expand
        
        if is_expanded:
            # Full expansion - generate Sequential with all nodes
            modules = [node.to_torch() for node in self.nodes]
            return f"nn.Sequential(\n    " + ",\n    ".join(modules) + "\n)"
        else:
            # Collapsed form - represent as a custom module
            return f"{self.type}()"  # Could include high-level params if needed
    
    def serialize(self) -> dict:
        """Serialize the sequence to a compact dict representation"""
        return {
            "type": self.type,
            "id": self.id,
            "collapsed": self.collapsed,
            "nodes": [self._serialize_node(node) for node in self.nodes],
            # Just store dimensions directly as a list (JSON-serializable)
            "in_shape": list(self.in_shape.dimensions) if self.in_shape else None,
            "out_shape": list(self.out_shape.dimensions) if self.out_shape else None,
        }
    
    @staticmethod
    def _serialize_node(node):          
        """Helper to serialize a node, handling both Node and Seq types"""
        if isinstance(node, Seq): 
            return node.serialize()
        # Simple serialization for basic nodes
        return {
            "id": node.id,
            "type": node.type,
            "params": node.params,
            # Direct dimension serialization
            "in_shape": list(node.in_shape.dimensions) if node.in_shape else None,
            "out_shape": list(node.out_shape.dimensions) if node.out_shape else None,
        }
    
    @classmethod
    def deserialize(cls, data: dict, node_registry=None) -> 'Seq':
        """
        Reconstruct a Seq from serialized data
        
        Args:
            data: Serialized sequence data
            node_registry: Optional dictionary mapping node IDs to existing Node instances
        """
        # Implementation depends on your node creation mechanism
        # This is a simplified example
        nodes = []
        for node_data in data["nodes"]:
            if "nodes" in node_data:  # This is a nested Seq
                nodes.append(Seq.deserialize(node_data, node_registry))
            else:
                # Create basic node
                node = Node(
                    type=node_data["type"],
                    params=node_data["params"]
                )
                nodes.append(node)
        
        # Direct Shape creation from dimensions list
        in_shape = Shape(data["in_shape"]) if data.get("in_shape") else None
        out_shape = Shape(data["out_shape"]) if data.get("out_shape") else None
        
        seq = cls(
            nodes=nodes,
            name=data["type"],
            collapsed=data.get("collapsed", False),
            in_shape=in_shape,
            out_shape=out_shape,
            id=data["id"]  # Pass ID directly in constructor
        )
        
        return seq


class Graph(): 
    def __init__(self, seqs: Optional[List[Seq]] = None, edges=None):
        """
        Initialize a graph using an adjacency list representation.
        
        Parameters:
        - seqs: Initial list of Seq objects to add to the graph, or None
        - edges: Initial list of edges as tuples (from_seq, operation_node, to_seq)
        - shape_match_check: If True, validates shape compatibility when adding edges
        """
        # Dictionary mapping sequences to list of (operation, destination) tuples
        self.seqs: Dict[Seq, List[Tuple[Node, Seq]]] = {}
        
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



