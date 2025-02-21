from node import Node
from typing import Tuple, Union
from enum import Enum

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

class ElementWiseNonlinearity(Node):
    def __init__(self, dim: Tuple[int, ...], nonlinearity: ElementWiseNonlinearityType = ElementWiseNonlinearityType.RELU):
        params = {'nonlinearity': nonlinearity}
        super().__init__(type="ElementWiseNonlinearity", params=params, in_shape=dim, out_shape=dim)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        # For element-wise operations, the output shape is the same as the input shape
        return in_shape

    def validate_params(self):
        # Ensure the nonlinearity type is valid
        if not isinstance(self.params['nonlinearity'], ElementWiseNonlinearityType):
            raise ValueError("Invalid nonlinearity type")

    def to_torch(self) -> str:
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
        return nonlinearity_map[self.params['nonlinearity']]


class Conv1D(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'padding_mode': padding_mode
        }
        super().__init__(type="Conv1D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        if 'in_channels' not in params or 'out_channels' not in params:
            raise ValueError("in_channels and out_channels must be specified")
        super().__init__(type="Conv1D", params=params)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, l = in_shape
        if c != self.params['in_channels']:
            raise ValueError("Input channels do not match")
        l_out = (l + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        return (n, self.params['out_channels'], l_out)

    def validate_params(self):
        if self.params['kernel_size'] <= 0:
            raise ValueError("Kernel size must be positive")
        if self.params['stride'] <= 0:
            raise ValueError("Stride must be positive")
        if self.params['padding'] < 0:
            raise ValueError("Padding cannot be negative")
        if self.params['dilation'] <= 0:
            raise ValueError("Dilation must be positive")
        if self.params['groups'] <= 0:
            raise ValueError("Groups must be positive")

    def to_torch(self) -> str:
        return f"nn.Conv1d({self.params['in_channels']}, {self.params['out_channels']}, {self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']}, dilation={self.params['dilation']}, groups={self.params['groups']}, bias={self.params['bias']}, padding_mode='{self.params['padding_mode']}')"


class Conv2D(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'padding_mode': padding_mode
        }
        super().__init__(type="Conv2D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        if 'in_channels' not in params or 'out_channels' not in params:
            raise ValueError("in_channels and out_channels must be specified")
        super().__init__(type="Conv2D", params=params)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, h, w = in_shape
        if c != self.params['in_channels']:
            raise ValueError("Input channels do not match")
        h_out = (h + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        w_out = (w + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        return (n, self.params['out_channels'], h_out, w_out)

    def validate_params(self):
        if self.params['kernel_size'] <= 0:
            raise ValueError("Kernel size must be positive")
        if self.params['stride'] <= 0:
            raise ValueError("Stride must be positive")
        if self.params['padding'] < 0:
            raise ValueError("Padding cannot be negative")
        if self.params['dilation'] <= 0:
            raise ValueError("Dilation must be positive")
        if self.params['groups'] <= 0:
            raise ValueError("Groups must be positive")

    def to_torch(self) -> str:
        return f"nn.Conv2d({self.params['in_channels']}, {self.params['out_channels']}, {self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']}, dilation={self.params['dilation']}, groups={self.params['groups']}, bias={self.params['bias']}, padding_mode='{self.params['padding_mode']}')"


class Conv3D(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'padding_mode': padding_mode
        }
        super().__init__(type="Conv3D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        if 'in_channels' not in params or 'out_channels' not in params:
            raise ValueError("in_channels and out_channels must be specified")
        super().__init__(type="Conv3D", params=params)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, d, h, w = in_shape
        if c != self.params['in_channels']:
            raise ValueError("Input channels do not match")
        d_out = (d + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        h_out = (h + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        w_out = (w + 2 * self.params['padding'] - self.params['dilation'] * (self.params['kernel_size'] - 1) - 1) // self.params['stride'] + 1
        return (n, self.params['out_channels'], d_out, h_out, w_out)

    def validate_params(self):
        if self.params['kernel_size'] <= 0:
            raise ValueError("Kernel size must be positive")
        if self.params['stride'] <= 0:
            raise ValueError("Stride must be positive")
        if self.params['padding'] < 0:
            raise ValueError("Padding cannot be negative")
        if self.params['dilation'] <= 0:
            raise ValueError("Dilation must be positive")
        if self.params['groups'] <= 0:
            raise ValueError("Groups must be positive")

    def to_torch(self) -> str:
        return f"nn.Conv3d({self.params['in_channels']}, {self.params['out_channels']}, {self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']}, dilation={self.params['dilation']}, groups={self.params['groups']}, bias={self.params['bias']}, padding_mode='{self.params['padding_mode']}')"


class PoolNode1D(Node):
    def __init__(self, kernel_size: int, stride=None, padding=0):
        params = {
            'kernel_size': kernel_size,
            'stride': stride if stride is not None else kernel_size,
            'padding': padding
        }
        super().__init__(type="PoolNode1D", params=params) 

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="PoolNode1D", params=params) 

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, l = in_shape
        l_out = (l + 2 * self.params['padding'] - self.params['kernel_size']) // self.params['stride'] + 1
        return (n, c, l_out)

    def validate_params(self):
        if self.params['kernel_size'] <= 0:
            raise ValueError("Kernel size must be positive")
        if self.params['stride'] <= 0:
            raise ValueError("Stride must be positive")
        if self.params['padding'] < 0:
            raise ValueError("Padding cannot be negative")

class MaxPool1D(PoolNode1D):
    def to_torch(self) -> str:
        return f"nn.MaxPool1d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class AvgPool1D(PoolNode1D):
    def to_torch(self) -> str:
        return f"nn.AvgPool1d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class LPPool1D(PoolNode1D):
    def __init__(self, norm_type: float, kernel_size: int, stride=None, padding=0):
        params = {
            'norm_type': norm_type, 
            'kernel_size': kernel_size,
            'stride': stride if stride is not None else kernel_size,
            'padding': padding
        }
        super().__init__(type="LPPool1d", params=params) 

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="LPPool1d", params=params) 

    def to_torch(self) -> str:
        return f"nn.LPPool1d(norm_type={self.params['norm_type']}, kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class PoolNode2D(Node):
    def __init__(self, kernel_size, stride=None, padding=0):
        params = {
            'kernel_size': (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            'stride': (stride, stride) if isinstance(stride, int) else stride or (kernel_size, kernel_size),
            'padding': (padding, padding) if isinstance(padding, int) else padding
        }
        super().__init__(type="PoolNode2D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="PoolNode2D", params=params)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, h, w = in_shape
        h_out = (h + 2 * self.params['padding'][0] - self.params['kernel_size'][0]) // self.params['stride'][0] + 1
        w_out = (w + 2 * self.params['padding'][1] - self.params['kernel_size'][1]) // self.params['stride'][1] + 1
        return (n, c, h_out, w_out)

    def validate_params(self):
        if any(k <= 0 for k in self.params['kernel_size']):
            raise ValueError("Kernel size must be positive")
        if any(s <= 0 for s in self.params['stride']):
            raise ValueError("Stride must be positive")
        if any(p < 0 for p in self.params['padding']):
            raise ValueError("Padding cannot be negative")

class MaxPool2D(PoolNode2D):
    def to_torch(self) -> str:
        return f"nn.MaxPool2d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class AvgPool2D(PoolNode2D):
    def to_torch(self) -> str:
        return f"nn.AvgPool2d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class LPPool2D(PoolNode2D):
    def __init__(self, norm_type: float, kernel_size, stride=None, padding=0):
        params = {
            'norm_type': norm_type,
            'kernel_size': (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            'stride': (stride, stride) if isinstance(stride, int) else stride or (kernel_size, kernel_size),
            'padding': (padding, padding) if isinstance(padding, int) else padding
        }
        super().__init__(type="LPPool2D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="LPPool2D", params=params)

    def to_torch(self) -> str:
        return f"nn.LPPool2d(norm_type={self.params['norm_type']}, kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class PoolNode3D(Node):
    def __init__(self, kernel_size, stride=None, padding=0):
        params = {
            'kernel_size': (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            'stride': (stride, stride, stride) if isinstance(stride, int) else stride or (kernel_size, kernel_size, kernel_size),
            'padding': (padding, padding, padding) if isinstance(padding, int) else padding
        }
        super().__init__(type="PoolNode3D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="PoolNode3D", params=params)

    def forward_dimension_inference(self, in_shape) -> Tuple[int, ...]:
        n, c, d, h, w = in_shape
        d_out = (d + 2 * self.params['padding'][0] - self.params['kernel_size'][0]) // self.params['stride'][0] + 1
        h_out = (h + 2 * self.params['padding'][1] - self.params['kernel_size'][1]) // self.params['stride'][1] + 1
        w_out = (w + 2 * self.params['padding'][2] - self.params['kernel_size'][2]) // self.params['stride'][2] + 1
        return (n, c, d_out, h_out, w_out)

    def validate_params(self):
        if any(k <= 0 for k in self.params['kernel_size']):
            raise ValueError("Kernel size must be positive")
        if any(s <= 0 for s in self.params['stride']):
            raise ValueError("Stride must be positive")
        if any(p < 0 for p in self.params['padding']):
            raise ValueError("Padding cannot be negative")

class MaxPool3D(PoolNode3D):
    def to_torch(self) -> str:
        return f"nn.MaxPool3d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class AvgPool3D(PoolNode3D):
    def to_torch(self) -> str:
        return f"nn.AvgPool3d(kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"

class LPPool3D(PoolNode3D):
    def __init__(self, norm_type: float, kernel_size, stride=None, padding=0):
        params = {
            'norm_type': norm_type,
            'kernel_size': (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            'stride': (stride, stride, stride) if isinstance(stride, int) else stride or (kernel_size, kernel_size, kernel_size),
            'padding': (padding, padding, padding) if isinstance(padding, int) else padding
        }
        super().__init__(type="LPPool3D", params=params)

    def __init__(self, params: Tuple[int, ...]):
        super().__init__(type="LPPool3D", params=params)

    def to_torch(self) -> str:
        return f"nn.LPPool3d(norm_type={self.params['norm_type']}, kernel_size={self.params['kernel_size']}, stride={self.params['stride']}, padding={self.params['padding']})"


