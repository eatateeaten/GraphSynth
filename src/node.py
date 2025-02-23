import torch
import abc
import uuid
from typing import Optional, Tuple, List
import tensorflow as tf 
import numpy as np 
from abc import ABC


class Node(abc.ABC):
    """
    Abstract Node class 
    An abstraction for all layers and aggregated NN-graph 
    In the form a Linked-list Node 
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
        for i in range(len(shape)): 
            if shape[i] <= 0:  
                raise InvalidShapeError(f"Invalid shape {shape}: {i}-th dim {shape[i]} must be larger than 0")
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
        Link a node to the input of this node, throw error if the other_node's out_shape is defined and mismatched with this node's in_shape
        """
        if self.in_shape is None:
            self.set_in_shape(other_node.out_shape)
        if self.in_shape != other_node.out_shape:
            raise InShapeMismatchError(f"Invalid add node, link node {self} to the output of {other_node}, {self} in_shape mismatch with {other_node} out_shape")
        self.input_node = other_node
        other_node.output_node = self 

    def set_output_node(self, other_node: 'Node'):
        """
        Link a node to the output of this node, throw error if the other_node's in_shape is defined and mismatched with this node's out_shape
        """
        if self.out_shape is None:
            self.set_out_shape(other_node.in_shape)
        if self.out_shape != other_node.in_shape:
            raise OutShapeMismatchError(f"Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        self.output_node = other_node 
        other_node.input_node = self

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
        self.validate_shape(new_in_shape)
        try:
            new_out_shape = self.forward_dimension_inference(new_in_shape)
            if self.out_shape is not None and new_out_shape != self.out_shape:
                raise InvalidShapeError(f"New in_shape {new_in_shape} produces out_shape {new_out_shape} that doesn't match existing out_shape {self.out_shape}")
            if self.out_shape is None: 
                self.validate_shape(new_out_shape)
                self.out_shape = new_out_shape 
        except (NotImplementedError, ValueError) as e:
            raise ForwardDimensionInferenceFailureError(f"Failed to infer out_shape from the new in_shape {new_in_shape}: {str(e)}")
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
        self.validate_shape(new_out_shape)
        if self.in_shape is None:
            self.out_shape = new_out_shape
        else: 
            try:
                out_shape = self.forward_dimension_inference(self.in_shape)
                if out_shape != new_out_shape:
                    raise InvalidShapeError(f"out_shape {new_out_shape} does not match the out_shape {out_shape} inferred from the existing in_shape {self.in_shape}")
            except (NotImplementedError, ValueError) as e:
                raise ForwardDimensionInferenceFailureError(f"While setting new out shape {new_out_shape}, failed to infer out_shape from the in_shape {self.in_shape}: {str(e)}. Check if the current in_shape {self.in_shape} and layer parameters are correct.")
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


class Tensor(Node):
    ## Tensor Type should be drawn with a different UI than other nodes 
    ## You should not allow people edit its input ouput shape
    """
    Represents a tensor node
    It's usually defined as an input and output node of a Seq 
    """
    def __init__(self, data: torch.Tensor| tf.Tensor | np.ndarray):
        super().__init__(type="Tensor", in_shape=data.shape , out_shape=data.shape) # The in_shape and out_shape of a Tensor type equals to its data shape. It is immutable and will never be None once the Node is defined. 
        self.data = data

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
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
        self.input_node = other_node
        other_node.output_node = self

    def set_output_node(self, other_node: 'Node'):
        """
        Overrides the Node Class method
        Link a node to the output of this node, throw error if the other_node's in_shape is defined and mismatched with this node's out_shape
        """
        if self.out_shape is None:
            raise ImmutableOutShapeError("Tensor out_shape should never be None.")
        if self.out_shape != other_node.in_shape:
            raise OutShapeMismatchError(f"Invalid add node, link node {self} to the input of {other_node}, {self} out_shape mismatch with {other_node} in_shape")
        self.output_node = other_node
        other_node.input_node = self


class Reshape(Node):
    ## I'm working to make Reshape and Swap_dim automatic, but for now make Reshape a specific node ig
    """
    Maps to reshape in Torch 
    """
    def __init__(self, out_dim):
        super().__init__(type="Reshape", params= {"out_dim": out_dim})

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
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
        return out_dim

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
