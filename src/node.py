import torch.nn as nn
from enum import Enum  
import torch
import abc
import uuid
from typing import Optional, Tuple, List

class POSITION(Enum):
    FRONT = 1   
    BACK = 2

class Node(abc.ABC):
    """
    Abstract Node class in a Linked-List 
    """
    def __init__(
        self,
        name: str = "",
        in_shape: Optional[Tuple[int, ...]] = None,
        out_shape: Optional[Tuple[int, ...]] = None,
        input_node: Optional["Node"] = None,
        output_node: Optional["Node"] = None,
    ): 
        self.id = str(uuid.uuid4())  # Unique ID
        self.name = name or f"Node_{self.id}"
        # Instead of in_dim/out_dim, we store a shape tuple
        self.in_shape: Optional[Tuple[int, ...]] = in_shape
        self.out_shape: Optional[Tuple[int, ...]] = out_shape

        if self.out_shape is None and self.in_shape is not None:
            try:
                self.out_shape = self.forward_dimension_inference()
            except (NotImplementedError, ValueError) as e:
                raise ValueError(f"Failed to infer out_shape: {str(e)}") 
            
        if self.in_shape is None and self.out_shape is not None:
            try:
                self.in_shape = self.backward_dimension_inference()
            except (NotImplementedError, ValueError) as e:
                raise ValueError(f"Failed to infer in_shape: {str(e)}")
            
        if self.in_shape is None and self.out_shape is None:
            raise ValueError(f"in_shape and out_shape cannot both be None")
            
        self.input_node: Optional["Node"] = input_node
        self.output_node: Optional["Node"] = output_node
    
    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Return a human-readable summary of the node (e.g. 'Linear((128,), (64,))').
        """
        pass


    @abc.abstractmethod
    def get_layer_type(self) -> str:
        """
        Return a string with the layer name.
        """
        pass

    @abc.abstractmethod
    def get_layer_params(self) -> dict:
        """
        Return a dict with the layer params.
        """
        pass
 
 
    @abc.abstractmethod
    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        """
        infer out_shape from in_shape with layer type and layer parameters
        """
        pass

    @abc.abstractmethod
    def backward_dimension_inference(self, out_shape) -> Tuple[int, ...]:
        """
        infer in_shape from out_shape with layer type and layer parameters
        """
        pass

    @abc.abstractmethod
    def check_shape_validity(self, shape: Tuple[int, ...]) -> str:
        """
        check if a shape is a valid tensor shape
        check shape against overflow, underflow, dividing-by-zero, etc. 
        """
        # check each dimension is larger than 0 
        for i in range(len(shape)): 
            if shape[i] <= 0:  
                raise ValueError("invalid shape {shape}: {i}-th dim {shape[i]} must be larger than 0") 
        return 
    
    @abc.abstractmethod
    def get_input_node(self) -> 'Node':
        """
        Return the input node of the current node
        """
        if self.input_node is None:
            raise ValueError("Input node is None")
        return self.input_node
    
    @abc.abstractmethod
    def get_output_node(self) -> 'Node':
        """
        Return the output node of the current node
        """
        if self.output_node is None:
            raise ValueError("Output node is None")
        return self.output_node

    @abc.abstractmethod
    def set_input_node(self, other_node: 'Node'):
        """
        Link a node to the input of this node 
        """
        if self.in_shape == None:
            self.set_in_shape(other_node.out_shape) # this call will check if with the new in_shape, the node is still valid 
        else:
            if self.in_shape != other_node.out_shape: # in_shape assigned and mismatched with prev out_shape
                raise ValueError("Invalid add node, link node {self} to the output of {other_node}, {self} in_shape mismatch with {other_node} out_shape")
        # all things check out, connect the two nodes
        self.input_node = other_node
        other_node.get_output_node = self # check correctness, check whether using a getter is a good way of doing things 

    @abc.abstractmethod        
    def set_output_node(self, other_node: 'Node'): 
        """
        Link a node to the output of this node 
        """
        if self.out_shape == None:   
            self.set_out_shape(other_node.in_shape) # this call will check if with the new out_shape, the node is still valid 
        else: 
            if self.out_shape != other_node.in_shape: # in_shape assigned and mismatched with prev node"s out_shape
                raise ValueError("Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        # all things check out, connect the two nodes
        self.output_node = other_node
        other_node.get_input_node = self # check correctness, check whether using a getter is a good way of doing things 
    # Note: connecting two nodes with the current node consists of two steps:
    # 1. adding the current node to the back of the prev node, calling set_input_node
    # 2. adding the current node to the front of the next node, calling set_output_node


    @abc.abstractmethod
    def set_in_shape(self, new_in_shape) -> 'Node':
        """
        Set the input shape.
        Then call the forward_dimension_inference 
        to check if the in_shape is valid for this layer type 
        and whether the resulting out_shape matches the predefined out_shape 
        """
        self.check_shape_validity(new_in_shape) ##TODO wrap it in a try so we can raise more meaningful errors
        try:
            new_out_shape = self.forward_dimension_inference(new_in_shape)
            if self.out_shape is not None and new_out_shape != self.out_shape:
                raise ValueError(f"New in_shape {new_in_shape} produces out_shape {new_out_shape} that doesn't match existing out_shape {self.out_shape}")
            if self.out_shape is None: 
                self.check_shape_validity(new_out_shape) ##TODO wrap it in a try so we can raise more meaningful errors
                self.out_shape = new_out_shape 
        except (NotImplementedError, ValueError) as e:
            raise ValueError(f"Failed to infer out_shape from the new in_shape {new_in_shape}: {str(e)}")
        self.in_shape = new_in_shape
        return self
    
    @abc.abstractmethod
    def set_out_shape(self, new_out_shape) -> 'Node':
        """
        Set the output shape.
        Then call the backward_dimension_inference 
        to check if the out_shape is valid for this layer type 
        and whether the resulting in_shape matches the predefined in_shape
        """
        self.check_shape_validity(new_out_shape) ##TODO wrap it in a try so we can raise more meaningful errors
        try:
            new_in_shape = self.backward_dimension_inference(new_out_shape)
            if self.in_shape is not None and new_in_shape != self.in_shape:
                raise ValueError(f"New out_shape {new_out_shape} requires in_shape {new_in_shape} that doesn't match existing in_shape {self.in_shape}")
            if self.out_shape is None: 
                self.check_shape_validity(new_in_shape) ##TODO wrap it in a try so we can raise more meaningful errors
                self.in_shape = new_in_shape 
        except (NotImplementedError, ValueError) as e:
            raise ValueError(f"Failed to infer in_shape from the new out_shape {new_out_shape}: {str(e)}")
        self.out_shape = new_out_shape 
        return self
    
    @abc.abstractmethod
    def to_torch(self) -> str:
        """
        Return a string with the PyTorch layer construction (e.g. 'nn.Linear(128, 64)').
        """
        pass
    

class PseudoNode(Node):
    """
    This Class exists for testing the functionalities of the Node Class 
    """
    


