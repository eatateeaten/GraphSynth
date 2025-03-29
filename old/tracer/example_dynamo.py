"""Example demonstrating how to use torch.compile with PyTorch Dynamo on a ResNet model."""

import time
import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import List, Dict, Any


def basic_compile_example():
    """Basic example of using torch.compile with a ResNet model."""
    print("\n" + "=" * 50)
    print("BASIC TORCH.COMPILE EXAMPLE WITH RESNET")
    print("=" * 50)
    
    # Create a ResNet18 model
    model = resnet18(weights=None)
    model.eval()
    
    # Create random input data
    example_input = torch.randn(1, 3, 224, 224)
    
    # Compile the model with default settings (backend='inductor')
    print("\nCompiling model with torch.compile()...")
    compiled_model = torch.compile(model)
    
    # First run will compile
    print("Running first inference (triggers compilation)...")
    start_time = time.time()
    output = compiled_model(example_input)
    first_run_time = time.time() - start_time
    print(f"First run took {first_run_time:.4f} seconds (includes compilation)")
    
    # Second run will use the compiled version
    print("Running second inference (uses compiled version)...")
    start_time = time.time()
    output = compiled_model(example_input)
    second_run_time = time.time() - start_time
    print(f"Second run took {second_run_time:.4f} seconds")
    
    # Run original model for comparison
    print("Running original model...")
    start_time = time.time()
    original_output = model(example_input)
    original_time = time.time() - start_time
    print(f"Original model took {original_time:.4f} seconds")
    
    # Verify outputs match
    torch.testing.assert_close(output, original_output)
    print("Outputs match between compiled and original model ✓")

    return compiled_model


def custom_compiler_backend():
    """Demonstrate using a custom compiler backend with torch.compile."""
    print("\n" + "=" * 50)
    print("CUSTOM COMPILER BACKEND WITH TORCH.COMPILE")
    print("=" * 50)
    
    # Create a ResNet18 model
    model = resnet18(weights=None)
    model.eval()
    
    # Create random input data
    example_input = torch.randn(1, 3, 224, 224)
    
    # Define a custom compiler backend
    def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """A simple custom backend that prints the FX graph and returns the original function."""
        print("\nCustom compiler called with FX graph:")
        gm.graph.print_tabular()
        
        # Count operations by type
        op_count = {}
        for node in gm.graph.nodes:
            op_type = node.op
            op_count[op_type] = op_count.get(op_type, 0) + 1
        
        print("\nOperation count by type:")
        for op_type, count in op_count.items():
            print(f"  {op_type}: {count}")
            
        # Just return the original function (no optimization in this example)
        return gm.forward
    
    # Compile the model with our custom backend
    print("\nCompiling model with custom backend...")
    compiled_model = torch.compile(model, backend=my_compiler)
    
    # Run the compiled model
    output = compiled_model(example_input)
    
    return compiled_model


def inspect_node_shapes():
    """Demonstrate how to get shape information from FX graph nodes."""
    print("\n" + "=" * 50)
    print("INSPECTING NODE SHAPES IN FX GRAPH")
    print("=" * 50)
    
    # Create a simple model with various operations to illustrate shape changes
    class ShapeExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 112 * 112, 10)
            
        def forward(self, x):
            x = self.conv(x)         # Shape change: [B, 3, 224, 224] -> [B, 16, 224, 224]
            x = torch.relu(x)        # Shape unchanged: [B, 16, 224, 224]
            x = self.pool(x)         # Shape change: [B, 16, 224, 224] -> [B, 16, 112, 112]
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)  # Shape change: [B, 16, 112, 112] -> [B, 16*112*112]
            x = self.fc(x)           # Shape change: [B, 16*112*112] -> [B, 10]
            return x
    
    # Create model and example input
    model = ShapeExampleModel()
    example_input = torch.randn(2, 3, 224, 224)  # Batch size 2
    
    # Method 1: Using FX symbolic_trace with ShapeProp
    print("\nMethod 1: Using torch.fx.symbolic_trace and ShapeProp")
    from torch.fx.passes.shape_prop import ShapeProp
    
    # Get the FX graph module through symbolic tracing
    fx_model = torch.fx.symbolic_trace(model)
    
    # Run shape propagation
    shape_prop = ShapeProp(fx_model)
    shape_prop.propagate(example_input)
    
    # Print node information including shapes
    print("\nNode shapes after propagation:")
    for node in fx_model.graph.nodes:
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            tensor_meta = node.meta['tensor_meta']
            shape = tensor_meta.shape if hasattr(tensor_meta, 'shape') else None
            dtype = tensor_meta.dtype if hasattr(tensor_meta, 'dtype') else None
            print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}")
            print(f"  Shape: {shape}, dtype: {dtype}")
        else:
            print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}")
            print(f"  Shape information not available")
    
    # Method 2: Using torch.compile with a custom backend
    print("\nMethod 2: Using torch.compile with custom backend")
    
    def shape_tracking_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """Custom backend that tracks shapes through the graph."""
        print("\nGraph module with shape information:")
        
        # Run shape propagation
        ShapeProp(gm).propagate(*example_inputs)
        
        # Print shape information
        for node in gm.graph.nodes:
            shape_str = "unknown"
            dtype_str = "unknown"
            
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                tensor_meta = node.meta['tensor_meta']
                if hasattr(tensor_meta, 'shape'):
                    shape_str = str(tensor_meta.shape)
                if hasattr(tensor_meta, 'dtype'):
                    dtype_str = str(tensor_meta.dtype)
            
            # For output nodes, look at the input to get its shape
            if node.op == 'output' and len(node.args) > 0:
                if isinstance(node.args[0], torch.fx.Node) and hasattr(node.args[0], 'meta'):
                    arg_meta = node.args[0].meta
                    if 'tensor_meta' in arg_meta:
                        shape_str = str(arg_meta['tensor_meta'].shape)
                        dtype_str = str(arg_meta['tensor_meta'].dtype)
            
            print(f"Node: {node.name}, Op: {node.op}, Target: {str(node.target)[:30]}{'...' if len(str(node.target)) > 30 else ''}")
            print(f"  Shape: {shape_str}, dtype: {dtype_str}")
            
            # Print input shapes too for call_function and call_module
            if node.op in ('call_function', 'call_method', 'call_module'):
                print("  Input shapes:")
                for i, arg in enumerate(node.args):
                    if isinstance(arg, torch.fx.Node) and hasattr(arg, 'meta'):
                        if 'tensor_meta' in arg.meta:
                            input_shape = arg.meta['tensor_meta'].shape
                            print(f"    Arg {i}: {input_shape}")
        
        # Return the original function
        return gm.forward
    
    # Compile the model with our shape-tracking backend
    compiled_model = torch.compile(model, backend=shape_tracking_compiler)
    
    # Run the model to trigger compilation
    output = compiled_model(example_input)
    
    # Method 3: Using fake tensors (FakeTensorMode)
    print("\nMethod 3: Using FakeTensor for shape tracing")
    try:
        from torch._dynamo.backends.common import fake_tensor_unsupported
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        # Create a mode that uses fake tensors (tensors that know their shape but don't allocate memory)
        with FakeTensorMode() as fake_mode:
            # Create fake input tensors with the same shape/dtype
            fake_input = fake_mode.from_tensor(example_input)
            
            # Trace with fake tensors
            traced_graph = torch.fx.trace(model, (fake_input,))
            
            print("\nNodes with shape information from fake tracing:")
            for node in traced_graph.nodes:
                if hasattr(node, 'meta') and 'val' in node.meta:
                    fake_val = node.meta['val']
                    if hasattr(fake_val, 'shape'):
                        print(f"Node: {node.name}, Shape: {fake_val.shape}, dtype: {fake_val.dtype}")
                    else:
                        print(f"Node: {node.name}, Non-tensor value: {type(fake_val)}")
                else:
                    print(f"Node: {node.name}, No shape information")
    except (ImportError, AttributeError) as e:
        print(f"FakeTensor method not available in this PyTorch version: {e}")
        
    return fx_model, compiled_model


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    
    # Basic example
    compiled_model = basic_compile_example()
    
    # Custom compiler backend
    custom_model = custom_compiler_backend()
    
    # Node shape inspection
    fx_model, shape_model = inspect_node_shapes()
    
    # Inspect Dynamo artifacts (if you've implemented this function)
    if 'inspect_dynamo_artifacts' in globals():
        inspected_model = inspect_dynamo_artifacts()
    
    # Different backend options (if you've implemented this function)
    if 'dynamo_with_backend_options' in globals():
        backend_results = dynamo_with_backend_options()
    
    print("\nAll examples completed successfully!")
