# Create a sequence of nodes using the provided Conv1DNode and other nodes
from layers_v1 import Conv1DNode, ElementWiseNonlinearity, FlattenNode, LinearNode, ElementWiseNonlinearityType


nodes = [
    Conv1DNode(batch_size=1, in_channels=3, out_channels=16, input_size=32, kernel_size=3, stride=1, padding=1),
    ElementWiseNonlinearity(dim=(1, 16, 32), nonlinearity=ElementWiseNonlinearityType.RELU),
    FlattenNode(dim = (1, 16, 32), start_dim = 1), 
    LinearNode(batch_size=1, input_features=16 * 32, output_features=10), 
    LinearNode(batch_size=1, input_features = 10, output_features=1)
]

seq = Seq(nodes)

# Generate PyTorch code
pytorch_code = seq.to_pytorch_code()
print(pytorch_code)
