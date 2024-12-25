import numpy as np 
from typing import List, Optional, Tuple

class Node:
    def __init__(self, in_dim: Tuple[int, ...], out_dim: Tuple[int, ...]):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name = f"Node_{id(self)}"
        self.input_nodes: List['Node'] = []
        self.output_nodes: List['Node'] = []

    def connect_to(self, other: 'Node') -> None:
        """Connect this node to another node."""
        if self.out_dim != other.in_dim:
            raise ValueError(
                f"Dimension mismatch: {self.name} output dim ({self.out_dim}) "
                f"!= {other.name} input dim ({other.in_dim})"
            )
           
        if other not in self.output_nodes:
            self.output_nodes.append(other)
        if self not in other.input_nodes:
            other.input_nodes.append(self)

class Graph:
    def __init__(self, in_node: Node, out_node: Node):
        self.nodes: List[Node] = []
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.completed: bool = False
        self.add_node(in_node)
        self.add_node(out_node)
    
    def add_node(self, node: Node, existing_node: Node, to_the_front: bool = False) -> None:
        if existing_node not in self.nodes:
            raise ValueError("The specified existing node is not part of the graph")
        if node not in self.nodes:
            # Check if the node can be connected to the specified existing node
            if node.in_dim == existing_node.out_dim or node.out_dim == existing_node.in_dim:
                self.nodes.append(node)
                if to_the_front:
                    self.connect_nodes(node, existing_node)
                else:
                    self.connect_nodes(existing_node, node)
            else:
                raise ValueError("Node dimensions do not match with the existing node for connection")
    
    def connect_nodes(self, from_node: Node, to_node: Node) -> None:
        """Connect two nodes and handle dimension mismatch errors."""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must be added to the graph before connecting")
        try:
            from_node.connect_to(to_node)
        except ValueError as e:
            raise e
