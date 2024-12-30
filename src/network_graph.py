import torch.nn as nn
from typing import List, Tuple, Union

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

    def forward(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

    def to_pytorch_code(self) -> str:
        raise NotImplementedError("Must be implemented by subclass.")

class ConvNode(Node):
    def __init__(self, batch_size, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        """
        Initializes a ConvNode representing a 2D convolutional layer.

        Parameters:
        - batch_size (int): Number of samples in a batch.
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - input_size (Tuple[int, int]): Spatial dimensions of the input (height, width).
        - kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel. Can be an int or a tuple (height, width).
        - stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Can be an int or a tuple (height, width). Default is 1.
        - padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input. Can be an int or a tuple (height, width). Default is 0.
        """
        # Ensure batch_size, in_channels, and out_channels are valid
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")

        # Ensure kernel_size is valid
        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError("kernel_size must be a positive integer")
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 2 or any(k <= 0 for k in kernel_size):
                raise ValueError("kernel_size must be a tuple of two positive integers")
        else:
            raise TypeError("kernel_size must be an int or a tuple of two ints")

        # Ensure stride is valid
        if isinstance(stride, int):
            if stride <= 0:
                raise ValueError("stride must be a positive integer")
            stride = (stride, stride)
        elif isinstance(stride, tuple):
            if len(stride) != 2 or any(s <= 0 for s in stride):
                raise ValueError("stride must be a tuple of two positive integers")
        else:
            raise TypeError("stride must be an int or a tuple of two ints")

        # Ensure padding is valid
        if isinstance(padding, int):
            if padding < 0:
                raise ValueError("padding must be a non-negative integer")
            padding = (padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) != 2 or any(p < 0 for p in padding):
                raise ValueError("padding must be a tuple of two non-negative integers")
        else:
            raise TypeError("padding must be an int or a tuple of two ints")

        # Check if the output dimensions are valid
        output_size = self.calculate_output_size(input_size, kernel_size, stride, padding)
        if any(o < 1 for o in output_size):
            raise ValueError("Invalid configuration: resulting output dimensions must be >= 1")

        super().__init__((batch_size, in_channels, *input_size), (batch_size, out_channels, *output_size))
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def calculate_output_size(self, input_size, kernel_size, stride, padding):
        # Calculate the output size for each dimension
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
        # Ensure batch_size, in_channels, and out_channels are valid
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")

        # Ensure kernel_size is valid
        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError("kernel_size must be a positive integer")
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            if len(kernel_size) != 3 or any(k <= 0 for k in kernel_size):
                raise ValueError("kernel_size must be a tuple of three positive integers")
        else:
            raise TypeError("kernel_size must be an int or a tuple of three ints")

        # Ensure stride is valid
        if isinstance(stride, int):
            if stride <= 0:
                raise ValueError("stride must be a positive integer")
            stride = (stride, stride, stride)
        elif isinstance(stride, tuple):
            if len(stride) != 3 or any(s <= 0 for s in stride):
                raise ValueError("stride must be a tuple of three positive integers")
        else:
            raise TypeError("stride must be an int or a tuple of three ints")

        # Ensure padding is valid
        if isinstance(padding, int):
            if padding < 0:
                raise ValueError("padding must be a non-negative integer")
            padding = (padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) != 3 or any(p < 0 for p in padding):
                raise ValueError("padding must be a tuple of three non-negative integers")
        else:
            raise TypeError("padding must be an int or a tuple of three ints")

        # Check if the output dimensions are valid
        output_size = self.calculate_output_size(input_size, kernel_size, stride, padding)
        if any(o < 1 for o in output_size):
            raise ValueError("Invalid configuration: resulting output dimensions must be >= 1")

        super().__init__((batch_size, in_channels, *input_size), (batch_size, out_channels, *output_size))
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def calculate_output_size(self, input_size, kernel_size, stride, padding):
        # Calculate the output size for each dimension
        return tuple(
            (input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
            for i in range(3)
        )

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Conv3d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding})"

class BatchNormNode(Node): 
    def __init__(self, num_features):
        super().__init__((num_features,), (num_features,))
        self.num_features = num_features

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.BatchNorm2d({self.num_features})"

class ReLUNode(Node):
    def __init__(self, dim: Tuple[int, ...]):
        super().__init__(dim, dim)

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return "nn.ReLU()"

class MaxPoolNode(Node):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__((None,), (None,))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.MaxPool2d({self.kernel_size}, stride={self.stride}, padding={self.padding})"

class DropoutNode(Node):
    def __init__(self, p=0.5):
        super().__init__((None,), (None,))
        self.p = p

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Dropout(p={self.p})"

class LinearNode(Node):
    def __init__(self, in_features, out_features):
        super().__init__((in_features,), (out_features,))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        pass

    def to_pytorch_code(self) -> str:
        return f"nn.Linear({self.in_features}, {self.out_features})"

class Graph:
    def __init__(self, in_node: Node, out_node: Node):
        self.nodes: List[Node] = []
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.completed: bool = False
        self.add_node(in_node)
        self.add_node(out_node)

    def add_node(self, node: Node, existing_node: Node = None, to_the_front: bool = False) -> None:
        if existing_node and existing_node not in self.nodes:
            raise ValueError("The specified existing node is not part of the graph")
        if node not in self.nodes:
            self.nodes.append(node)
            if existing_node:
                if node.in_dim == existing_node.out_dim or node.out_dim == existing_node.in_dim:
                    if to_the_front:
                        self.connect_nodes(node, existing_node)
                    else:
                        self.connect_nodes(existing_node, node)
                else:
                    raise ValueError("Node dimensions do not match with the existing node for connection")

    def connect_nodes(self, from_node: Node, to_node: Node) -> None:
        try:
            from_node.connect_to(to_node)
        except ValueError as e:
            raise e

    def to_pytorch_code(self) -> str:
        code_lines = []
        for node in self.nodes:
            code_lines.append(node.to_pytorch_code())
            # sequential = nn.Sequential(*modules)
        return "\n".join(code_lines)

# Example usage
conv_node = ConvNode(batch_size=32, in_channels=3, out_channels=16, input_size=(32, 32), kernel_size=3)
batch_norm_node = BatchNormNode(num_features=16)
relu_node = ReLUNode(dim=(16,))
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
