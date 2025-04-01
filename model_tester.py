#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# Import from other test files
from test_neural_net import (
    save_generated_model,
    load_model_from_file,
    visualize_model_torchviz,
    visualize_model_hiddenlayer,
    visualize_model_netron,
    test_model_forward
)

from test_complex_graph import (
    trace_through_model,
    compare_models
)

def read_model_from_file(file_path):
    """Read model code from a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading model file: {e}")
        return None

def parse_input_size(size_str):
    """Parse input size string to tuple or list of tuples"""
    try:
        # Try to interpret as a single input size
        parts = [int(x.strip()) for x in size_str.split(',')]
        if len(parts) >= 2:
            return tuple(parts)
        else:
            raise ValueError("Input size must have at least 2 dimensions")
    except Exception as e:
        print(f"Error parsing input size: {e}")
        print("Using default size (1, 3, 224, 224)")
        return (1, 3, 224, 224)

def main():
    parser = argparse.ArgumentParser(description='Neural Network Testing Tool')
    
    # Model input arguments
    parser.add_argument('--model', '-m', type=str, help='Path to the Python file containing model code')
    parser.add_argument('--code', '-c', type=str, help='Direct model code string (use with care)')
    parser.add_argument('--input-size', '-i', type=str, default="1,3,224,224", 
                        help='Input tensor size as comma-separated values (e.g., "1,3,224,224")')
    
    # Visualization options
    parser.add_argument('--viz-all', '-a', action='store_true', help='Enable all visualizations')
    parser.add_argument('--viz-torchviz', '-t', action='store_true', help='Enable torchviz visualization')
    parser.add_argument('--viz-hiddenlayer', '-l', action='store_true', help='Enable hiddenlayer visualization')
    parser.add_argument('--viz-netron', '-n', action='store_true', help='Enable netron visualization')
    parser.add_argument('--viz-feature-maps', '-f', action='store_true', help='Enable feature map visualization')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default="nn_model", help='Base name for output files')
    
    args = parser.parse_args()
    
    # Get model code
    model_code = None
    if args.model:
        model_code = read_model_from_file(args.model)
    elif args.code:
        model_code = args.code
    else:
        print("No model provided. Please specify either --model or --code.")
        parser.print_help()
        return
    
    if not model_code:
        print("Failed to get model code. Exiting.")
        return
    
    # Parse input size
    input_size = parse_input_size(args.input_size)
    
    # Save model to file
    model_file = save_generated_model(model_code, f"{args.output}.py")
    
    # Load model
    model = load_model_from_file(model_file)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Test model forward pass
    print("\nTesting forward pass...")
    test_model_forward(model, input_size)
    
    # Visualizations
    if args.viz_all or args.viz_torchviz:
        print("\nGenerating torchviz visualization...")
        visualize_model_torchviz(model, input_size, f"{args.output}_graph")
    
    if args.viz_all or args.viz_hiddenlayer:
        print("\nGenerating hiddenlayer visualization...")
        try:
            visualize_model_hiddenlayer(model, input_size, f"{args.output}_hiddenlayer")
        except Exception as e:
            print(f"Error with hiddenlayer visualization: {e}")
    
    if args.viz_all or args.viz_netron:
        print("\nGenerating netron visualization...")
        try:
            visualize_model_netron(model, input_size, f"{args.output}_netron.onnx")
        except Exception as e:
            print(f"Error with Netron visualization: {e}")
    
    if args.viz_all or args.viz_feature_maps:
        print("\nGenerating feature map visualizations...")
        try:
            input_tensor = torch.randn(input_size)
            trace_through_model(model, input_tensor)
        except Exception as e:
            print(f"Error with feature map visualization: {e}")
    
    print(f"\nAll outputs saved with base name: {args.output}")

if __name__ == "__main__":
    main() 