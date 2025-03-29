"""Example demonstrating how to trace PyTorch models using torch.fx."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torchvision.models import resnet18

def trace_library_resnet():
    """Trace a ResNet model from torchvision library."""
    print("\n" + "=" * 50)
    print("TRACING TORCHVISION RESNET MODEL")
    print("=" * 50)

    # Get a pretrained ResNet18 model
    print("\nLoading pretrained ResNet18 from torchvision...")
    resnet = resnet18(pretrained=False)
    resnet.eval()  # Set to evaluation mode
    
    # Create example input for ResNet
    example_input = torch.randn(1, 3, 224, 224)
    
    # Print model structure summary
    print("\nResNet18 Structure:")
    print(resnet)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in resnet.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Attempt to trace with symbolic_trace
    print("\nTracing ResNet18 with torch.fx.symbolic_trace...")
    traced_model = fx.symbolic_trace(resnet)
    
    # Print the traced graph summary
    print("\nGraph IR Summary:")
    node_count = len(list(traced_model.graph.nodes))
    print(f"Total nodes in graph: {node_count}")
    
    # Print a sample of the graph (first 10 nodes)
    print("\nSample of Graph IR (first 10 nodes):")
    for i, node in enumerate(traced_model.graph.nodes):
        if i < 10:
            print(f"  {node}")
        else:
            break
            
    print("\n... (graph truncated)")
    
    # Print a sample of the generated code
    print("\nSample of Generated Python Code:")
    code_lines = traced_model.code.split('\n')
    for i, line in enumerate(code_lines):
        if i < 15:
            print(line)
        else:
            break
            
    print("... (code truncated)")
    
    return traced_model

if __name__ == "__main__":
    # Trace a library ResNet model
    traced_resnet = trace_library_resnet()
