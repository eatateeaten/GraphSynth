import torch.nn as nn
from typing import List, Tuple, Union
from enum import Enum  
import torch

class Seq():
    """
    The unit for model code generation. Generate "nn.Sequantial" - like model code. 
    All nodes forming a Seq are serialized into a compressed representation
    Seq only works with Regular Nodes. It does not work with MergeNode and BranchNode. 
    Supports Training Code Generation. 
    """ 
    def __init__(self, start_node, end_node): 
        self.in_shape = Tuple[int, ...]
        self.out_shape = Tuple[int, ...]
        start_node = Node 
        end_node = Node 

    def to_torch(self): 
        return 

    def serialize_from_nodes(self): 
        return 

    def parse_to_nodes(self):
        return 


class Train_loop():
    """
    The unit for model code generation. Generate "nn.Sequantial"-like model code. 
    All nodes forming a Seq are serialized into a compressed representation
    Only works with Regular Nodes.
    Does not work with MergeNode and BranchNode. 
    Supports Training Code Generation. 
    """
    in_shape = Tuple[int, ...]
    out_shape = Tuple[int, ...]

    def to_torch(): 
        
        return 

    def serialize_from_nodes(): 
        return 

    def parse_to_nodes():
        return 


class NN_Graph():
    """
    Combines Seqs and MergeNode, BranchNode to cover all Neural Network Architecture Cases 
    Training Code Generation 
    """


#A seq cannot start off with a node that needs to infer its in_dim (input dimension) from the previous node
class Seq:
    def __init__(self, nodes: List[Node]):
        """
        Initialize a sequence with a list of nodes.

        Parameters:
        - nodes (List[Node]): A list of nodes forming the sequence
        """
        if not nodes:
            raise ValueError("The sequence must contain at least one node.")
        
        self._validate_dimensions(nodes)
        self.nodes = nodes
        self.in_node = nodes[0]
        self.out_node = nodes[-1]
        self._rename_nodes()

    def _validate_dimensions(self, nodes: List[Node]) -> None:
        """
        Validate that each node's out_dim matches the next node's in_dim.

        Parameters:
        - nodes (List[Node]): A list of nodes to validate
        """
        for i in range(len(nodes) - 1):
            if nodes[i].out_dim != nodes[i + 1].in_dim:
                raise ValueError(
                    f"Dimension mismatch: {nodes[i].name} output dim ({nodes[i].out_dim}) "
                    f"!= {nodes[i + 1].name} input dim ({nodes[i + 1].in_dim})"
                )

    def _rename_nodes(self):
        seq_id = id(self)
        for index, node in enumerate(self.nodes):
            node.name = f"Seq{seq_id}_{node.__class__.__name__}_{index}"

    def connect_to(self, to_seq: 'Seq') -> None:
        """
        Connect this sequence to another sequence.  
        Parameters:
        - to_seq (Seq): Target sequence to connect to
        """
        if self.out_node.out_dim != to_seq.in_node.in_dim:
            raise ValueError(
                f"Dimension mismatch: {self.out_node.name} output dim ({self.out_node.out_dim}) "
                f"!= {to_seq.in_node.name} input dim ({to_seq.in_node.in_dim})"
            )

        # Connect the sequences
        self.out_node.output_node = to_seq.in_node
        to_seq.in_node.input_node = self.out_node

        # Merge the sequences
        self.nodes.extend(to_seq.nodes)

    def connect_from(self, from_seq: 'Seq') -> None:
        """
        Connect this sequence from another sequence.

        Parameters:
        - from_seq (Seq): Source sequence to connect from
        """
        if from_seq.out_node.out_dim != self.in_node.in_dim:
            raise ValueError(
                f"Dimension mismatch: {from_seq.out_node.name} output dim ({from_seq.out_node.out_dim}) "
                f"!= {self.in_node.name} input dim ({self.in_node.in_dim})"
            )

        # Connect the sequences
        from_seq.out_node.output_node = self.in_node
        self.in_node.input_node = from_seq.out_node

        # Merge the sequences
        self.nodes = from_seq.nodes + self.nodes

    def to_pytorch_code(self) -> str:
        """
        Generate PyTorch code for the sequence.

        Returns:
        - str: PyTorch code representing the sequence
        """
        module_calls = [node.to_pytorch_code() for node in self.nodes]
        module_list = ",\n            ".join(module_calls)
