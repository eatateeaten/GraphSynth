import torch
import uuid
from typing import Optional, Tuple, List, Dict, Union
import tensorflow as tf 
import numpy as np 
from abc import ABC
from .shape import Shape, InvalidShapeError
from .torch_nn_module_calls import validate_params, get_torch_code

class Node:
    """
    An abstraction for all layers and aggregated NN-seq 
    """
    def __init__(
        self,
        type: str,
        params: dict = {},
        id: str = None,                     #ID assigned from front-end 
        in_shape: Optional[Shape] = None, 
        out_shape: Optional[Shape] = None, 
        in_node: Optional["Node"] = None,  
        out_node: Optional["Node"] = None,  
    ):
        self.id = id or str(uuid.uuid4())       # If not assigned an ID from front-end, generate an ID 
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
    
    # def completed(self):
    #    return self.in_shape is not None and self.out_shape is not None and self.in_node is not None and self.out_node is not None 
   
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



class Seq(Node):
    """
    A sequence of nodes that can be treated as a single node or expanded into its components.
    Supports collapsible/expandable representation for UI and code generation.
    """
    def __init__(
        self, 
        nodes: List[Node] = None,
        name: str = None,  # Descriptive name for collapsed view (e.g., "ResNet Block")
        collapsed: bool = False,  # Default display state
        id: str = None,                     #ID assigned from front-end 
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



