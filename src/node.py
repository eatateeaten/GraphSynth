import torch.nn as nn
from enum import Enum  
import torch
import abc
import uuid
from typing import Optional, Tuple, List


#Some notes to front-end: 
#the bubble sould only look green if a node has in_shape, out_shape, input_node and output_node all defined
#call completed() to check 

class Node(abc.ABC):
    """
    Abstract Node class in a Linked-List 
    """
    def __init__(
        self,
        type: str = "",
        params: dict = {},
        in_shape: Optional[Tuple[int, ...]] = None,
        out_shape: Optional[Tuple[int, ...]] = None,
        input_node: Optional["Node"] = None,
        output_node: Optional["Node"] = None,
    ):
        self.id = str(uuid.uuid4())  # Unique ID
        self.type= type or f"Node_{self.id}"
        self.params = params
        self.in_shape = in_shape
        self.out_shape = out_shape

        if self.out_shape is None and self.in_shape is not None:
            try:
                self.out_shape = self.forward_dimension_inference()
            except (NotImplementedError, ValueError) as e:
                raise ValueError(f"Failed to infer out_shape: {str(e)}") 
            
        # if self.in_shape is None and self.out_shape is not None:
        #     try:
        #         self.in_shape = self.backward_dimension_inference()
        #     except (NotImplementedError, ValueError) as e:
        #         raise ValueError(f"Failed to infer in_shape: {str(e)}")
            
        #if self.in_shape is None and self.out_shape is None:
        #    raise ValueError(f"in_shape and out_shape cannot both be None")
            
        self.input_node: Optional["Node"] = input_node
        self.output_node: Optional["Node"] = output_node
    
    
    def __str__(self) -> str:
        """
        Return a human-readable summary of the node (e.g. 'Linear((128,), (64,))').
        """
        params_str = ", ".join(f"{param}={val}" for param, val in self.params.items())
        return f"{self.type}({params_str})"
    
    def completed(self):
        return self.in_shape != None and self.out_shape != None and self.input_node != None and self.output_node != None 
    

    def get_layer_type(self) -> str:
        """
        Return a string with the layer name.
        """
        return self.type

    
    def get_layer_params(self) -> dict:
        """
        Return a dict with the layer params.
        """
        return self.params 
    
    def validate_shape(self, shape: Tuple[int, ...]) -> str:
        """
        check if a shape is a valid tensor shape
        check shape against overflow, underflow, dividing-by-zero, etc. 
        """
        # check each dimension is larger than 0 
        for i in range(len(shape)): 
            if shape[i] <= 0:  
                raise ValueError("invalid shape {shape}: {i}-th dim {shape[i]} must be larger than 0") 
        return 
    
    
    def get_input_node(self) -> 'Node':
        """
        Return the input node of the current node
        """
        if self.input_node is None:
            raise ValueError("Input node is None")
        return self.input_node
    
    
    def get_output_node(self) -> 'Node':
        """
        Return the output node of the current node
        """
        if self.output_node is None:
            raise ValueError("Output node is None")
        return self.output_node


    def set_input_node(self, other_node: 'Node'):
        """
        Link a node to the input of this node, throw error if the other_node's out_shape is defined and mismatch with this node's in_shape
        """
        if self.in_shape == None:
            self.set_in_shape(other_node.out_shape) # this call will check if with the new in_shape, the node is still valid 
        else:
            if self.in_shape != other_node.out_shape: # in_shape assigned and mismatched with prev out_shape
                raise ValueError("Invalid add node, link node {self} to the output of {other_node}, {self} in_shape mismatch with {other_node} out_shape")
        # all things check out, connect the two nodes
        self.input_node = other_node
        other_node.output_node = self 
        #because we already checked for the validity of linking other_node to the input. we can directly set the output_node of 
        #other node to this node 
    

    def set_output_node(self, other_node: 'Node'): 
        """
        Link a node to the output of this node, throw error if the other_node's in_shape is defined and mismatch with this node's out_shape
        """
        if self.out_shape == None:   
            self.set_out_shape(other_node.in_shape) # this call will check if with the new out_shape, the node is still valid 
        else: 
            if self.out_shape != other_node.in_shape: # in_shape assigned and mismatched with prev node"s out_shape
                raise ValueError("Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        # all things check out, connect the two nodes
        self.output_node = other_node 
        other_node.input_node = self 
        #because we already checked for the validity of linking other_node to the output. we can directly set the input_node of 
        #other node to this node 

    # Note: connecting two nodes with the current node consists of two steps:
    # 1. adding the current node to the back of the prev node, calling set_input_node
    # 2. adding the current node to the front of the next node, calling set_output_node

    def set_in_shape(self, new_in_shape) -> 'Node':
        """
        Set the input shape.
        Then call the forward_dimension_inference 
        to check if the in_shape is valid for this layer type 
        and whether the resulting out_shape matches the predefined out_shape 
        """
        self.validate_shape(new_in_shape) ##TODO wrap it in a try so we can raise more meaningful errors
        try:
            new_out_shape = self.forward_dimension_inference(new_in_shape)
            if self.out_shape is not None and new_out_shape != self.out_shape:
                raise ValueError(f"New in_shape {new_in_shape} produces out_shape {new_out_shape} that doesn't match existing out_shape {self.out_shape}")
            if self.out_shape is None: 
                self.validate_shape(new_out_shape) ##TODO wrap it in a try so we can raise more meaningful errors
                self.out_shape = new_out_shape 
        except (NotImplementedError, ValueError) as e:
            raise ValueError(f"Failed to infer out_shape from the new in_shape {new_in_shape}: {str(e)}")
        self.in_shape = new_in_shape
        return 
     
    
    def set_out_shape(self, new_out_shape) -> 'Node':
        """
        Set the output shape.
        Only check if the out_shape is valid 
        Does not apply dimensional inference to retrieve the in-shape 
        As in many cases with nn.modules such as Conv2D and tensor operations such as Reshape backward dimension inference is not impossible 
        TODO The out_shape can be set if in-shape is None. 
        When we set in-shape in the set_in_shape call in the future we will check if the defined out_shape matches the inferred out_shape 
        TODO If in_shape is already defined, we will do a forward_dimension_inference to check if it is valid. 
        """
        self.validate_shape(new_out_shape) ##TODO wrap it in a try so we can raise more meaningful errors
        if self.in_shape is None:
            self.out_shape = new_out_shape
        else: 
            try:
                out_shape = self.forward_dimension_inference(self.in_shape)
                if out_shape != new_out_shape:
                # This error is not supposed to be reached, we put it here for robustness 
                    raise ValueError(f"out_shape {new_out_shape} does not match the out_shape {out_shape} inferred from the existing in_shape{self.in_shape}")
            except (NotImplementedError, ValueError) as e:
                # This error is not supposed to be reached, we put it here for robustness 
                raise ValueError(f"While setting new out shape {new_out_shape}, failed to infer out_shape from the in_shape {self.in_shape}: {str(e)}. Check if the current in_shape{self.in_shape} and layer parameters are correct.")
            self.out_shape = new_out_shape
        return 
    
    

    @abc.abstractmethod
    def validate_params(self):
        """
        validate the layer parameters with requirements specific to each layer type 
        """
        pass
    
    @abc.abstractmethod
    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        """
        infer out_shape from in_shape with layer type and layer parameters
        cfcf vb TODO  in some cases, the layer parameters already defined the out_shape, like Reshape 
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


class Reshape(Node):
    """
    Maps to reshape in Torch 
    """
    def __init__(self, out_dim = [-1, ]):
        super().__init__(type="Reshape", params= {"out_dim": [-1, ]})

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        factor = self.params['factor']
        return (in_shape[0], in_shape[1] * factor) + in_shape[factor:]

    def backward_dimension_inference(self, out_shape) -> Tuple[int, ...]:
        factor = self.params['factor']
        if len(out_shape) < factor or out_shape[1] % factor != 0:
            raise ValueError("The second dimension must be at least {factor} and divisible by {factor}")
        return (out_shape[0], out_shape[1] // factor) + out_shape[factor:]

    def to_torch(self) -> str:
        return "PseudoLayer_second_dim_multiply()"


class PseudoNode_second_dim_multiply(Node):
    """
    This Class exists for testing the functionalities of the Node Class 
    """
    def __init__(self, factor=1):
        super().__init__(type="PseudoNode_second_dim_multiply", params={'factor': factor})

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        factor = self.params['factor']
        return (in_shape[0], in_shape[1] * factor) + in_shape[factor:]

    def backward_dimension_inference(self, out_shape) -> Tuple[int, ...]:
        factor = self.params['factor']
        if len(out_shape) < factor or out_shape[1] % factor != 0:
            raise ValueError("The second dimension must be at least {factor} and divisible by {factor}")
        return (out_shape[0], out_shape[1] // factor) + out_shape[factor:]

    def to_torch(self) -> str:
        return "PseudoLayer_second_dim_multiply()"


class PseudoNode_second_dim_divide(Node):
    """
    This Class exists for testing the functionalities of the Node Class 
    """
    def __init__(self, factor=1):
        super().__init__(type="PseudoNode_second_dim_divide", params={'factor': factor}) 

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        factor = self.params['factor'] 
        if len(in_shape) < factor or in_shape[1] % factor != 0:
            raise ValueError("The second dimension must be even and at least {factor} and be divisible by {factor}")
        return (in_shape[0], in_shape[1] / factor) + in_shape[factor:]

    def backward_dimension_inference(self, out_shape) -> Tuple[int, ...]:
        factor = self.params['factor'] 
        return (out_shape[0], out_shape[1] * self.params['factor']) + out_shape[factor:]

    def to_torch(self) -> str:
        return "PseudoLayer_second_dim_multiply()"
