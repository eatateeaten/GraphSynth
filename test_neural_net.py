import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz
import netron
import os
import sys
import tempfile

# Import common utilities
from test_utils import save_generated_model, load_model_from_file

# Try to import hiddenlayer (might not be available in all environments)
try:
    import hiddenlayer as hl
except ImportError:
    print("hiddenlayer not available. Some visualizations will be disabled.")
    hl = None

def visualize_model_torchviz(model, input_size, filename="model_graph"):
    """Visualize model using torchviz"""
    # Create example input based on the expected input size
    if isinstance(input_size, tuple):
        # Single input
        dummy_input = torch.randn(input_size)
        inputs = (dummy_input,)
    elif isinstance(input_size, list):
        # Multiple inputs
        inputs = tuple(torch.randn(size) for size in input_size)
    else:
        raise ValueError("input_size must be a tuple or list of tuples")
    
    # Run the model once to get the graph
    try:
        output = model(*inputs)
        
        # Create a dot graph
        dot = torchviz.make_dot(output, params=dict(model.named_parameters()))
        
        # Save and show the graph
        dot.format = 'png'
        dot.render(filename, cleanup=True)
        print(f"Graph visualization saved to {filename}.png")
    except Exception as e:
        print(f"Error visualizing model with torchviz: {e}")

def visualize_model_hiddenlayer(model, input_size, filename="model_hiddenlayer"):
    """Visualize model using hiddenlayer if available"""
    if hl is None:
        print("hiddenlayer not available. Skipping visualization.")
        return
    
    # Create example input based on the expected input size
    if isinstance(input_size, tuple):
        # Single input
        dummy_input = torch.randn(input_size)
        inputs = [dummy_input]
    elif isinstance(input_size, list):
        # Multiple inputs
        inputs = [torch.randn(size) for size in input_size]
    else:
        raise ValueError("input_size must be a tuple or list of tuples")
    
    try:
        # Create visualization
        graph = hl.build_graph(model, inputs)
        
        # Save the visualization
        graph.save(f"{filename}.png")
        print(f"HiddenLayer visualization saved to {filename}.png")
    except Exception as e:
        print(f"Error visualizing model with hiddenlayer: {e}")

def visualize_model_netron(model, input_size, filename="model_netron.pt"):
    """Save model to file and open with netron"""
    # Create example input based on the expected input size
    if isinstance(input_size, tuple):
        # Single input
        dummy_input = torch.randn(input_size)
        inputs = (dummy_input,)
    elif isinstance(input_size, list):
        # Multiple inputs
        inputs = tuple(torch.randn(size) for size in input_size)
    else:
        raise ValueError("input_size must be a tuple or list of tuples")
    
    try:
        # Save the model
        torch.onnx.export(model, inputs, filename)
        print(f"Model saved to {filename} for Netron visualization")
        
        # Launch Netron to visualize the model
        netron_port = 8080
        netron.start(filename, port=netron_port)
        print(f"Netron visualization server started at http://localhost:{netron_port}")
    except Exception as e:
        print(f"Error saving model for Netron: {e}")

def test_model_forward(model, input_size):
    """Test if the model runs forward pass successfully"""
    # Create example input based on the expected input size
    if isinstance(input_size, tuple):
        # Single input
        dummy_input = torch.randn(input_size)
        inputs = (dummy_input,)
    elif isinstance(input_size, list):
        # Multiple inputs
        inputs = tuple(torch.randn(size) for size in input_size)
    else:
        raise ValueError("input_size must be a tuple or list of tuples")
    
    try:
        # Run the model
        with torch.no_grad():
            output = model(*inputs)
        
        # Print output information
        if isinstance(output, tuple):
            print("Model produced multiple outputs:")
            for i, out in enumerate(output):
                print(f"  Output {i}: shape={out.shape}")
        else:
            print(f"Model produced single output with shape: {output.shape}")
        
        return output
    except Exception as e:
        print(f"Error running model forward pass: {e}")
        return None

def main():
    # Example usage
    print("Neural Network Visualization Tool")
    print("================================")
    
    # Example model code (replace with your generated code)
    example_model_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class NNBlock(nn.Module):
    def __init__(self):
        super(NNBlock, self).__init__()
        # Empty __init__ for now
        pass
        
    def forward(self, input_0):
        var_0 = torch.split(input_0, sections=(1,1), dim=1)
        var_1, var_2 = torch.split(input_0, sections=(1,1), dim=1)
        return var_1, var_2
"""

    # Save the model code to a file
    model_file = save_generated_model(example_model_code)
    
    # Load the model
    model = load_model_from_file(model_file)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Set the input size (adjust based on your model's expected input)
    # For a single input tensor with batch_size=1, channels=2, height=224, width=224
    input_size = (1, 2, 224, 224)
    
    # Test forward pass
    test_model_forward(model, input_size)
    
    # Visualize the model using different methods
    visualize_model_torchviz(model, input_size)
    visualize_model_hiddenlayer(model, input_size)
    visualize_model_netron(model, input_size)
    
    print("\nTo use with your generated model:")
    print("1. Replace the example_model_code with your generated code")
    print("2. Adjust input_size to match your model's expected input")
    print("3. Run this script again")

if __name__ == "__main__":
    main() 