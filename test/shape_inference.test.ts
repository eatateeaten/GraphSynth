import { forwardShapeInference } from '../isomorphic/torch_nn_module_op';
import { execSync } from 'child_process';
import { join } from 'path';

// Helper function to generate random shapes within reasonable bounds
function generateRandomShape(dims: number): number[] {
    return Array(dims).fill(0).map((_, i) => {
        if (i === 0) return Math.floor(Math.random() * 32) + 1; // batch size
        if (i === 1) return Math.floor(Math.random() * 64) + 1; // channels
        return Math.floor(Math.random() * 128) + 8; // spatial dimensions
    });
}

// Helper function to run PyTorch shape test
function runPyTorchShapeTest(moduleType: string, inputShape: number[], params: Record<string, any>): number[] {
    let pythonScript = `
import torch
import torch.nn.functional as F

# Create random input
input_tensor = torch.randn(${inputShape.join(', ')})

# Apply layer
`;

    // Add module-specific PyTorch code
    switch (moduleType) {
        case 'Linear':
            pythonScript += `
output = F.linear(input_tensor, torch.randn(${params.output_features}, ${params.input_features}), bias=torch.randn(${params.output_features}) if ${params.bias ? "True" : "False"} else None)
`;
            break;
        case 'Conv2D':
            pythonScript += `
output = F.conv2d(input_tensor, 
                                 torch.randn(${params.out_channels}, ${params.in_channels}, ${params.kernel_size}, ${params.kernel_size}),
                                 bias=torch.randn(${params.out_channels}) if ${params.bias ? "True" : "False"} else None,
                                 stride=${params.stride},
                                 padding=${params.padding},
                                 dilation=${params.dilation},
                                 groups=${params.groups})
`;
            break;
        case 'ReLU':
            pythonScript += `
output = F.relu(input_tensor, inplace=${params.inplace ? "True" : "False"})
`;
            break;
        case 'BatchNorm2D':
            pythonScript += `
output = F.batch_norm(input_tensor, 
                                         torch.randn(${params.num_features}),
                                         torch.randn(${params.num_features}),
                                         torch.randn(${params.num_features}),
                                         torch.randn(${params.num_features}),
                                         training=True,
                                         momentum=${params.momentum},
                                         eps=${params.eps})
`;
            break;
        case 'MaxPool2D':
            pythonScript += `
output = F.max_pool2d(input_tensor,
                                         kernel_size=${params.kernel_size},
                                         stride=${params.stride},
                                         padding=${params.padding},
                                         dilation=${params.dilation},
                                         ceil_mode=${params.ceil_mode ? "True" : "False"})
`;
            break;
        case 'Dropout2D':
            pythonScript += `
output = F.dropout2d(input_tensor,
                                        p=${params.p},
                                        training=True,
                                        inplace=${params.inplace ? "True" : "False"})
`;
            break;
        case 'Reshape':
            pythonScript += `
output = input_tensor.reshape(${params.shape.join(', ')})
`;
            break;
        default:
            throw new Error(`Unsupported module type: ${moduleType}`);
    }

    pythonScript += `
print(output.shape)
`;

    // Write script to temporary file
    const scriptPath = join(__dirname, 'temp_shape_test.py');
    require('fs').writeFileSync(scriptPath, pythonScript);

    // Run Python script and capture output
    const output = execSync(`python3 ${scriptPath}`).toString().trim();
    
    // Clean up
    require('fs').unlinkSync(scriptPath);

    // Parse output shape - handle torch.Size([...]) format
    const shapeStr = output.replace('torch.Size(', '').replace(')', '');
    return shapeStr.slice(1, -1).split(', ').map(Number);
}

describe('Shape Inference Tests', () => {
    // Linear
    test('Linear shape inference matches PyTorch', () => {
        const inputShape = [2, 512];
        const params = {
            input_features: 512,
            output_features: 256,
            bias: true
        };

        const ourShape = forwardShapeInference('Linear', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('Linear', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Convolutional
    test('Conv2D shape inference matches PyTorch', () => {
        const inputShape = [2, 3, 32, 32];
        const params = {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            bias: true
        };

        const ourShape = forwardShapeInference('Conv2D', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('Conv2D', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Activation
    test('ReLU shape inference matches PyTorch', () => {
        const inputShape = [2, 3, 32, 32];
        const params = {
            inplace: false
        };

        const ourShape = forwardShapeInference('ReLU', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('ReLU', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Reshape
    test('Reshape shape inference matches PyTorch', () => {
        const inputShape = [2, 64, 32, 32];
        const params = {
            shape: [2, 64, 1024]
        };

        const ourShape = forwardShapeInference('Reshape', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('Reshape', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Pooling
    test('MaxPool2D shape inference matches PyTorch', () => {
        const inputShape = [2, 64, 32, 32];
        const params = {
            kernel_size: 2,
            stride: 2,
            padding: 0,
            dilation: 1,
            ceil_mode: false
        };

        const ourShape = forwardShapeInference('MaxPool2D', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('MaxPool2D', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Normalization
    test('BatchNorm2D shape inference matches PyTorch', () => {
        const inputShape = [2, 64, 32, 32];
        const params = {
            num_features: 64,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true
        };

        const ourShape = forwardShapeInference('BatchNorm2D', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('BatchNorm2D', inputShape, params);

        expect(ourShape).toEqual(pytorchShape);
    });

    // Dropout
    test('Dropout2D shape inference matches PyTorch', () => {
        const inputShape = [2, 64, 32, 32];
        const params = {
            p: 0.5,
            inplace: false
        };

        const ourShape = forwardShapeInference('Dropout2D', inputShape, params);
        const pytorchShape = runPyTorchShapeTest('Dropout2D', inputShape, params);
        
        expect(ourShape).toEqual(pytorchShape);
    });
});
