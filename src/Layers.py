    
from network_seq import Node 
from typing import Tuple, Union
from enum import Enum

class FlattenNode(Node):
    def __init__(self, dim, start_dim: int = 1, end_dim = -1): 
        """
        Initializes a FlattenNode representing a flatten operation. Asumming (B, ..) format 

        Parameters:
        - start_dim (int): The first dimension to flatten. Default is 1.
        - end_dim (int): The last dimension to flatten. Default is -1.
        """
        
        self.start_dim = start_dim
        self.end_dim = end_dim if end_dim != -1 else len(dim)
        super().__init__(dim, self.calculate_output_size(dim, start_dim, self.end_dim)) 
        self.name = f"Flatten_{id(self)}"

    def calculate_output_size(self, dim, start_dim, end_dim):
        flattened_dim = 1
        for d in dim[start_dim:end_dim]:
            flattened_dim *= d

         # Construct the out_dim
        out_dim = dim[:start_dim] + (flattened_dim,) + dim[end_dim:]
        return out_dim
    
    def to_pytorch_code(self) -> str:
        return f"nn.Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


class LinearNode(Node):
    def __init__(self, batch_size: int, input_features: int, output_features: int):
        """
        Initializes a LinearNode representing a fully-connected layer. 

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
        self.name = f"Linear_{id(self)}"

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
        self.name = f"ElementWiseNonlinearity_{id(self)}"

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
        self.name = f"Nonlinearity1D_{id(self)}"

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
        self.name = f"Conv1D_{id(self)}"

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
        self.name = f"Conv2D_{id(self)}"

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
        self.name = f"Conv3D_{id(self)}"

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
        self.name = f"PoolNode1D_{id(self)}"

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
        self.name = f"LPPool1D_{id(self)}"

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
        self.name = f"PoolNode2D_{id(self)}"

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
        self.name = f"LPPool2D_{id(self)}"

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
        self.name = f"PoolNode3D_{id(self)}"

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
        self.name = f"LPPool3D_{id(self)}"

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
        self.name = f"AdaptivePool1D_{id(self)}"

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
        self.name = f"AdaptivePool2D_{id(self)}"

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
        self.name = f"AdaptivePool3D_{id(self)}"

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
    
