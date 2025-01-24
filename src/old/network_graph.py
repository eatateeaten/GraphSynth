import torch.nn as nn
from typing import List, Tuple, Union
from enum import Enum


class Node:
    def __init__(self, in_dim: Tuple[int, ...], out_dim: Tuple[int, ...]):
        """
        Initialize a Node with input and output dimensions.

        Parameters:
        - in_dim (Tuple[int, ...]): Input dimensions
        - out_dim (Tuple[int, ...]): Output dimensions
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name = f"Node_{id(self)}"
        self.input_node: 'Node' = None
        self.output_node: 'Node' = None

    def forward(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class Seq:
    def __init__(self, in_node: Node, out_node: Node):
        """
        Initialize a sequence with input and output nodes.

        Parameters:
        - in_node (Node): The first node in the sequence
        - out_node (Node): The last node in the sequence
        """
        self.nodes: List[Node] = [in_node, out_node]
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.completed: bool = False

    def connect_to(self, node: Node, to_seq: 'Seq') -> None:
        """
        Connect this sequence to another sequence through a new node.
        The node will be connected after this sequence's out_node.

        Parameters:
        - node (Node): Node to connect with
        - to_seq (Seq): Target sequence to connect to
        """
        if node.in_dim != self.out_node.out_dim:
            raise ValueError(
                f"Dimension mismatch: {self.out_node.name} output dim ({self.out_node.out_dim}) "
                f"!= {node.name} input dim ({node.in_dim})"
            )
        
        if self.out_node.output_node is not None:
            raise ValueError(f"{self.out_node.name} already has an output node")
        if node.input_node is not None:
            raise ValueError(f"{node.name} already has an input node")

        # Connect the nodes
        self.out_node.output_node = node
        node.input_node = self.out_node

        # Update the sequences
        self.nodes.insert(-1, node)
        to_seq.nodes.insert(1, node)

    def connect_from(self, node: Node, from_seq: 'Seq') -> None:
        """
        Connect this sequence from another sequence through a new node.
        The node will be connected before this sequence's in_node.

        Parameters:
        - node (Node): Node to connect with
        - from_seq (Seq): Source sequence to connect from
        """
        if node.out_dim != self.in_node.in_dim:
            raise ValueError(
                f"Dimension mismatch: {node.name} output dim ({node.out_dim}) "
                f"!= {self.in_node.name} input dim ({self.in_node.in_dim})"
            )
        
        if node.output_node is not None:
            raise ValueError(f"{node.name} already has an output node")
        if self.in_node.input_node is not None:
            raise ValueError(f"{self.in_node.name} already has an input node")

        # Connect the nodes
        node.output_node = self.in_node
        self.in_node.input_node = node

        # Update the sequences
        self.nodes.insert(0, node)
        from_seq.nodes.insert(-1, node)

    def to_pytorch_code(self) -> str:
        """
        Generate PyTorch code for the sequence.

        Returns:
        - str: PyTorch code representing the sequence
        """
        code_lines = [node.to_pytorch_code() for node in self.nodes]
        return "\n".join(code_lines)



class LinearNode(Node):
    def __init__(self, batch_size: int, input_features: int, output_features: int):
        """
        Initializes a LinearNode representing a fully connected layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - input_features (int): Number of input features.
        - output_features (int): Number of output features.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(input_features, int) and input_features > 0, "input_features must be a positive integer"
        assert isinstance(output_features, int) and output_features > 0, "output_features must be a positive integer"

        super().__init__((batch_size, input_features), (batch_size, output_features))
        self.batch_size = batch_size
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Linear({self.input_features}, {self.output_features})"


class ElementWiseNonlinearityType(Enum):
    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    LEAKY_RELU = "LeakyReLU"
    ELU = "ELU"
    SELU = "SELU"
    CELU = "CELU"
    GELU = "GELU"
    SOFTPLUS = "Softplus"
    SOFTSIGN = "Softsign"
    HARDTANH = "Hardtanh"
    HARDSHRINK = "Hardshrink"
    HARDSIGMOID = "Hardsigmoid"
    HARDSWISH = "Hardswish"
    SOFTSHRINK = "Softshrink"
    TANHSHRINK = "Tanhshrink"
    THRESHOLD = "Threshold"
    RELU6 = "ReLU6"
    SILU = "SiLU"
    MISH = "Mish"

class Nonlinearity1DType(Enum):
    SOFTMAX = "Softmax"
    LOG_SOFTMAX = "LogSoftmax"
    GLU = "GLU"

class ElementWiseNonlinearity(Node):
    def __init__(self, dim: Tuple[int, ...], nonlinearity: ElementWiseNonlinearityType = ElementWiseNonlinearityType.RELU):
        """
        Initializes an ElementWiseNonlinearity node representing an element-wise nonlinearity layer.

        Parameters:
        - dim (Tuple[int, ...]): The dimensions of the input and output.
        - nonlinearity (ElementWiseNonlinearityType, optional): The type of nonlinearity to apply. Default is ElementWiseNonlinearityType.RELU.
        """
        super().__init__(dim, dim)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        nonlinearity_map = {
            ElementWiseNonlinearityType.RELU: "nn.ReLU()",
            ElementWiseNonlinearityType.SIGMOID: "nn.Sigmoid()",
            ElementWiseNonlinearityType.TANH: "nn.Tanh()",
            ElementWiseNonlinearityType.LEAKY_RELU: "nn.LeakyReLU()",
            ElementWiseNonlinearityType.ELU: "nn.ELU()",
            ElementWiseNonlinearityType.SELU: "nn.SELU()",
            ElementWiseNonlinearityType.CELU: "nn.CELU()",
            ElementWiseNonlinearityType.GELU: "nn.GELU()",
            ElementWiseNonlinearityType.SOFTPLUS: "nn.Softplus()",
            ElementWiseNonlinearityType.SOFTSIGN: "nn.Softsign()",
            ElementWiseNonlinearityType.HARDTANH: "nn.Hardtanh()",
            ElementWiseNonlinearityType.HARDSHRINK: "nn.Hardshrink()",
            ElementWiseNonlinearityType.HARDSIGMOID: "nn.Hardsigmoid()",
            ElementWiseNonlinearityType.HARDSWISH: "nn.Hardswish()",
            ElementWiseNonlinearityType.SOFTSHRINK: "nn.Softshrink()",
            ElementWiseNonlinearityType.TANHSHRINK: "nn.Tanhshrink()",
            ElementWiseNonlinearityType.THRESHOLD: "nn.Threshold(0, 0)",  # Example threshold
            ElementWiseNonlinearityType.RELU6: "nn.ReLU6()",
            ElementWiseNonlinearityType.SILU: "nn.SiLU()",
            ElementWiseNonlinearityType.MISH: "nn.Mish()"
        }

        return nonlinearity_map[self.nonlinearity]

class Nonlinearity1D(Node):
    def __init__(self, dim: Tuple[int, ...], nonlinearity: Nonlinearity1DType, dim_index: int = -1):
        """
        Initializes a Nonlinearity1D node representing a nonlinearity layer that operates across a single dimension.

        Parameters:
        - dim (Tuple[int, ...]): The dimensions of the input and output.
        - nonlinearity (Nonlinearity1DType): The type of nonlinearity to apply.
        - dim_index (int, optional): The dimension index to apply the nonlinearity. Default is -1 (last dimension).s
        """
        super().__init__(dim, dim)
        self.nonlinearity = nonlinearity
        self.dim_index = dim_index

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        nonlinearity_map = {
            Nonlinearity1DType.SOFTMAX: f"nn.Softmax(dim={self.dim_index})",
            Nonlinearity1DType.LOG_SOFTMAX: f"nn.LogSoftmax(dim={self.dim_index})",
            Nonlinearity1DType.GLU: f"nn.GLU(dim={self.dim_index})"
        }

        return nonlinearity_map[self.nonlinearity]
    
class Conv1DNode(Node):
    def __init__(self, batch_size, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        ### batch_size, in_channels, input_size, should be inferred, the rest are given by user 
        """
        Initializes a Conv1DNode representing a 1D convolutional layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - input_size (int): Length of the input sequence.
        - kernel_size (Union[int, Tuple[int]]): Size of the convolutional kernel. Can be an int or a tuple (length,).
        - stride (Union[int, Tuple[int]], optional): Stride of the convolution. Can be an int or a tuple (length,). Default is 1.
        - padding (Union[int, Tuple[int]], optional): Padding added to both sides of the input. Can be an int or a tuple (length,). Default is 0.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(in_channels, int) and in_channels > 0, "in_channels must be a positive integer"
        assert isinstance(out_channels, int) and out_channels > 0, "out_channels must be a positive integer"

        kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 1 and all(k > 0 for k in kernel_size), \
            "kernel_size must be a positive integer or a tuple of one positive integer"

        stride = (stride,) if isinstance(stride, int) else stride
        assert isinstance(stride, tuple) and len(stride) == 1 and all(s > 0 for s in stride), \
            "stride must be a positive integer or a tuple of one positive integer"

        padding = (padding,) if isinstance(padding, int) else padding
        assert isinstance(padding, tuple) and len(padding) == 1 and all(p >= 0 for p in padding), \
            "padding must be a non-negative integer or a tuple of one non-negative integer"

        output_size = self.calculate_output_size(input_size, kernel_size, stride, padding)
        assert output_size >= 1, "Invalid configuration: resulting output dimension must be >= 1"

        super().__init__((batch_size, in_channels, input_size), (batch_size, out_channels, output_size))
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def calculate_output_size(self, input_size, kernel_size, stride, padding):
        return (input_size + 2 * padding[0] - kernel_size[0]) // stride[0] + 1

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Conv1d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding})"

class Conv2DNode(Node):
    def __init__(self, batch_size, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        """
        Initializes a Conv2DNode representing a 2D convolutional layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - input_size (Tuple[int, int]): Spatial dimensions of the input (height, width).
        - kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel. Can be an int or a tuple (height, width).
        - stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Can be an int or a tuple (height, width). Default is 1.
        - padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input. Can be an int or a tuple (height, width). Default is 0.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(in_channels, int) and in_channels > 0, "in_channels must be a positive integer"
        assert isinstance(out_channels, int) and out_channels > 0, "out_channels must be a positive integer"

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2 and all(k > 0 for k in kernel_size), \
            "kernel_size must be a positive integer or a tuple of two positive integers"

        stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, tuple) and len(stride) == 2 and all(s > 0 for s in stride), \
            "stride must be a positive integer or a tuple of two positive integers"

        padding = (padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, tuple) and len(padding) == 2 and all(p >= 0 for p in padding), \
            "padding must be a non-negative integer or a tuple of two non-negative integers"

        output_size = self.calculate_output_size(input_size, kernel_size, stride, padding)
        assert all(o >= 1 for o in output_size), "Invalid configuration: resulting output dimensions must be >= 1"

        super().__init__((batch_size, in_channels, *input_size), (batch_size, out_channels, *output_size))
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def calculate_output_size(self, input_size, kernel_size, stride, padding):
        return tuple(
            (input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
            for i in range(2)
        )

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Conv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding})"

class Conv3DNode(Node):
    def __init__(self, batch_size, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        """
        Initializes a Conv3DNode representing a 3D convolutional layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - input_size (Tuple[int, int, int]): Spatial dimensions of the input (depth, height, width).
        - kernel_size (Union[int, Tuple[int, int, int]]): Size of the convolutional kernel. Can be an int or a tuple (depth, height, width).
        - stride (Union[int, Tuple[int, int, int]], optional): Stride of the convolution. Can be an int or a tuple (depth, height, width). Default is 1.
        - padding (Union[int, Tuple[int, int, int]], optional): Padding added to all sides of the input. Can be an int or a tuple (depth, height, width). Default is 0.
        """
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
        assert isinstance(in_channels, int) and in_channels > 0, "in_channels must be a positive integer"
        assert isinstance(out_channels, int) and out_channels > 0, "out_channels must be a positive integer"

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 3 and all(k > 0 for k in kernel_size), \
            "kernel_size must be a positive integer or a tuple of three positive integers"

        stride = (stride, stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, tuple) and len(stride) == 3 and all(s > 0 for s in stride), \
            "stride must be a positive integer or a tuple of three positive integers"

        padding = (padding, padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, tuple) and len(padding) == 3 and all(p >= 0 for p in padding), \
            "padding must be a non-negative integer or a tuple of three non-negative integers"

        output_size = self.calculate_output_size(input_size, kernel_size, stride, padding)
        assert all(o >= 1 for o in output_size), "Invalid configuration: resulting output dimensions must be >= 1"

        super().__init__((batch_size, in_channels, *input_size), (batch_size, out_channels, *output_size))
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding

    def calculate_output_size(self, input_size, kernel_size, stride, padding):
        return tuple(
            (input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
            for i in range(3)
        )

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Conv3d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding})"




# BatchNorm 

# Dropout 



##Pool 


class PoolNode1D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: int, kernel_size: int, stride=None, padding=0):
        """
        Initializes a PoolNode1D representing a 1D pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (int): Length of the input sequence.
        - kernel_size (int): Size of the pooling window.
        - stride (int, optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (int, optional): Padding added to both sides of the input. Default is 0.
        """
        assert isinstance(kernel_size, int) and kernel_size > 0, "kernel_size must be a positive integer" 
        assert stride is None or (isinstance(stride, int) and stride > 0), "stride must be a positive integer or None"
        assert isinstance(padding, int) and padding >= 0, "padding must be a non-negative integer"

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        output_size = self.calculate_output_size(input_size)
        assert output_size > 0, (
            "Pooled output size must be positive. Please ensure that "
            "(input_size + 2 * self.padding - self.kernel_size) // self.stride + 1 > 0"
        )

        super().__init__((batch_size, in_channels, input_size), (batch_size, in_channels, output_size))

    def calculate_output_size(self, input_size: int) -> int:
        """
        Calculates the output size of the pooling layer.

        Parameters:
        - input_size (int): The size of the input.

        Returns:
        - int: The size of the output.
        """
        return (input_size + 2 * self.padding - self.kernel_size) // self.stride + 1

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class MaxPool1D(PoolNode1D):
    def to_pytorch_code(self) -> str:
        return f"nn.MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool1D(PoolNode1D):
    def to_pytorch_code(self) -> str:
        return f"nn.AvgPool1d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class LPPool1D(PoolNode1D):
    def __init__(self, batch_size: int, in_channels: int, input_size: int, norm_type: float, kernel_size: int, stride=None, padding=0):
        """
        Initializes an LPPool1D representing a 1D Lp pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (int): Length of the input sequence.
        - norm_type (float): The norm type for the pooling operation.
        - kernel_size (int): Size of the pooling window.
        - stride (int, optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (int, optional): Padding added to both sides of the input. Default is 0.
        """
        assert isinstance(norm_type, (int, float)) and norm_type > 0, "norm_type must be a positive number"
        super().__init__(batch_size, in_channels, input_size, kernel_size, stride, padding)
        self.norm_type = norm_type

    def to_pytorch_code(self) -> str:
        return f"nn.LPPool1d(norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class PoolNode2D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int], kernel_size, stride=None, padding=0):
        """
        Initializes a PoolNode2D representing a 2D pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int]): Height and width of the input.
        - kernel_size (Union[int, Tuple[int, int]]): Size of the pooling window.
        - stride (Union[int, Tuple[int, int]], optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input. Default is 0.
        """
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        ##TODO double check if PyTorch allows the pooling window to not be square 
        ##if not then maybe we have to enforce kernal_size to be int? 
        self.stride = (stride, stride) if isinstance(stride, int) else stride or self.kernel_size
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        output_size = self.calculate_output_dimension(input_size)
        assert all(o > 0 for o in output_size), (
            "Pooled output dimensions must be positive. Please ensure that "
            "each dimension satisfies (input_size[i] + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1 > 0"
        )

        super().__init__((batch_size, in_channels, *input_size), (batch_size, in_channels, *output_size))

    def calculate_output_dimension(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(
            (input_size[i] + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1
            for i in range(2)
        )

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class MaxPool2D(PoolNode2D):
    def to_pytorch_code(self) -> str:
        return f"nn.MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2D(PoolNode2D):
    def to_pytorch_code(self) -> str:
        return f"nn.AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class LPPool2D(PoolNode2D):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int], norm_type: float, kernel_size, stride=None, padding=0):
        """
        Initializes an LPPool2D representing a 2D Lp pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int]): Height and width of the input.
        - norm_type (float): The norm type for the pooling operation.
        - kernel_size (Union[int, Tuple[int, int]]): Size of the pooling window.
        - stride (Union[int, Tuple[int, int]], optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input. Default is 0.
        """
        assert isinstance(norm_type, (int, float)) and norm_type > 0, "norm_type must be a positive number"
        super().__init__(batch_size, in_channels, input_size, kernel_size, stride, padding)
        self.norm_type = norm_type

    def to_pytorch_code(self) -> str:
        return f"nn.LPPool2d(norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class PoolNode3D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int, int], kernel_size, stride=None, padding=0):
        """
        Initializes a PoolNode3D representing a 3D pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int, int]): Depth, height, and width of the input.
        - kernel_size (Union[int, Tuple[int, int, int]]): Size of the pooling window.
        - stride (Union[int, Tuple[int, int, int]], optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (Union[int, Tuple[int, int, int]], optional): Padding added to all sides of the input. Default is 0.
        """
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride or self.kernel_size
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding

        output_size = self.calculate_output_dimension(input_size)
        assert all(o > 0 for o in output_size), (
            "Pooled output dimensions must be positive. Please ensure that "
            "each dimension satisfies (input_size[i] + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1 > 0"
        )

        super().__init__((batch_size, in_channels, *input_size), (batch_size, in_channels, *output_size))

    def calculate_output_dimension(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(
            (input_size[i] + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1
            for i in range(3)
        )

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class MaxPool3D(PoolNode3D):
    def to_pytorch_code(self) -> str:
        return f"nn.MaxPool3d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool3D(PoolNode3D):
    def to_pytorch_code(self) -> str:
        return f"nn.AvgPool3d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class LPPool3D(PoolNode3D):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int, int], norm_type: float, kernel_size, stride=None, padding=0):
        """
        Initializes an LPPool3D representing a 3D Lp pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int, int]): Depth, height, and width of the input.
        - norm_type (float): The norm type for the pooling operation.
        - kernel_size (Union[int, Tuple[int, int, int]]): Size of the pooling window.
        - stride (Union[int, Tuple[int, int, int]], optional): Stride of the pooling operation. Default is None, which means it will be set to kernel_size.
        - padding (Union[int, Tuple[int, int, int]], optional): Padding added to all sides of the input. Default is 0.
        """
        assert isinstance(norm_type, (int, float)) and norm_type > 0, "norm_type must be a positive number"
        super().__init__(batch_size, in_channels, input_size, kernel_size, stride, padding)
        self.norm_type = norm_type

    def to_pytorch_code(self) -> str:
        return f"nn.LPPool3d(norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")



#AdaptivePool 
class AdaptivePool1D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: int, output_size: int):
        """
        Initializes an AdaptivePool1D node representing a 1D adaptive pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (int): Length of the input sequence.
        - output_size (int): Desired length of the output sequence.
        """
        assert isinstance(output_size, int) and output_size > 0, "output_size must be a positive integer"
        assert output_size <= input_size, "output_size must be less than or equal to input_size"

        super().__init__((batch_size, in_channels, input_size), (batch_size, in_channels, output_size))
        self.output_size = output_size

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class AdaptivePool2D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int], output_size: Tuple[int, int]):
        """
        Initializes an AdaptivePool2D node representing a 2D adaptive pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int]): Height and width of the input.
        - output_size (Tuple[int, int]): Desired height and width of the output.
        """
        assert isinstance(output_size, tuple) and len(output_size) == 2 and all(isinstance(o, int) and o > 0 for o in output_size), \
            "output_size must be a tuple of two positive integers"
        assert all(o <= i for o, i in zip(output_size, input_size)), "output_size must be less than or equal to input_size in each dimension"

        super().__init__((batch_size, in_channels, *input_size), (batch_size, in_channels, *output_size))
        self.output_size = output_size

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")


class AdaptivePool3D(Node):
    def __init__(self, batch_size: int, in_channels: int, input_size: Tuple[int, int, int], output_size: Tuple[int, int, int]):
        """
        Initializes an AdaptivePool3D node representing a 3D adaptive pooling layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - input_size (Tuple[int, int, int]): Depth, height, and width of the input.
        - output_size (Tuple[int, int, int]): Desired depth, height, and width of the output.
        """
        assert isinstance(output_size, tuple) and len(output_size) == 3 and all(isinstance(o, int) and o > 0 for o in output_size), \
            "output_size must be a tuple of three positive integers"
        assert all(o <= i for o, i in zip(output_size, input_size)), "output_size must be less than or equal to input_size in each dimension"

        super().__init__((batch_size, in_channels, *input_size), (batch_size, in_channels, *output_size))
        self.output_size = output_size

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")

class AdaptiveMaxPool1D(AdaptivePool1D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveMaxPool1d(output_size={self.output_size})"


class AdaptiveAveragePool1D(AdaptivePool1D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveAvgPool1d(output_size={self.output_size})"


class AdaptiveMaxPool2D(AdaptivePool2D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveMaxPool2d(output_size={self.output_size})"


class AdaptiveAveragePool2D(AdaptivePool2D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveAvgPool2d(output_size={self.output_size})"


class AdaptiveMaxPool3D(AdaptivePool3D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveMaxPool3d(output_size={self.output_size})"


class AdaptiveAveragePool3D(AdaptivePool3D):
    def to_pytorch_code(self) -> str:
        return f"nn.AdaptiveAvgPool3d(output_size={self.output_size})"
    

# Example usage
conv_node = ConvNode(batch_size=32, in_channels=3, out_channels=16, input_size=(32, 32), kernel_size=3)
batch_norm_node = BatchNormNode(num_features=16)
relu_node = ElementWiseNonlinearity(dim=(16,))
max_pool_node = MaxPoolNode(kernel_size=2)
dropout_node = DropoutNode(p=0.5)
linear_node = LinearNode(in_features=16, out_features=10)

graph = Graph(in_node=conv_node, out_node=linear_node)
graph.add_node(batch_norm_node, existing_node=conv_node)
graph.add_node(relu_node, existing_node=batch_norm_node)
graph.add_node(max_pool_node, existing_node=relu_node)
graph.add_node(dropout_node, existing_node=max_pool_node)

# Serialize to PyTorch code
pytorch_code = graph.to_pytorch_code()
print(pytorch_code) 