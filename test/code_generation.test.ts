import { Graph } from '../isomorphic/graph';
import { v4 as uuidv4 } from 'uuid';

describe('Code Generation Tests', () => {
    test('Simple CNN graph generates correct PyTorch code', () => {
        // Create a new graph
        const graph = new Graph();

        // Create input tensor
        const inputId = uuidv4();
        graph.createPendingNode('Tensor', inputId, {
            shape: [1, 3, 32, 32],
            target: 'torch',
            variableName: 'x'
        });

        // Create Conv2D layer
        const convId = uuidv4();
        graph.createPendingNode('Op', convId, {
            opType: 'Conv2D',
            target: 'torch',
            opParams: {
                in_channels: 3,
                out_channels: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                bias: true
            }
        });

        // Create ReLU layer
        const reluId = uuidv4();
        graph.createPendingNode('Op', reluId, {
            opType: 'ReLU',
            target: 'torch',
            opParams: {
                inplace: false
            }
        });

        // Create MaxPool2D layer
        const poolId = uuidv4();
        graph.createPendingNode('Op', poolId, {
            opType: 'MaxPool2D',
            target: 'torch',
            opParams: {
                kernel_size: 2,
                stride: 2,
                padding: 0
            }
        });

        // Create output tensor
        const outputId = uuidv4();
        graph.createPendingNode('Tensor', outputId, {
            shape: [1, 64, 16, 16], // Shape after Conv2D -> ReLU -> MaxPool2D
            target: 'torch',
            variableName: 'output'
        });

        // Make input tensor a source
        graph.makeTensorSource(inputId);

        // Connect the layers
        graph.connect(inputId, convId);
        graph.connect(convId, reluId);
        graph.connect(reluId, poolId);
        graph.connect(poolId, outputId);

        // Get the generated code
        const code = graph.to_torch_functional();
        console.log(code);

        // Expected PyTorch code
        const expectedCode = `import torch
import torch.nn.functional as F

def forward(x):
    var_0 = F.conv2d(x, torch.randn(64, 3, 3, 3), bias=torch.randn(64), stride=1, padding=1)
    var_1 = F.relu(var_0, inplace=False)
    var_2 = F.max_pool2d(var_1, kernel_size=2, stride=2, padding=0)
    output = var_2
    return output`;

        // Compare the generated code with expected code
        expect(code.trim()).toBe(expectedCode.trim());
    });
});
