#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# Import common utilities
from test_utils import save_generated_model, load_model_from_file, read_model_from_file, parse_input_size
from test_neural_net import (
    visualize_model_torchviz,
    visualize_model_hiddenlayer, 
    visualize_model_netron,
    test_model_forward
)
from test_complex_graph import (
    trace_through_model,
    compare_models
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Network Model Tester')
    
    # Model input options (exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', help='Path to Python file with model code')
    model_group.add_argument('--code', help='Direct model code input')
    
    # Visualization options
    parser.add_argument('--input-size', default='1,3,224,224', help='Input tensor size as comma-separated values (e.g. 1,3,224,224)')
    parser.add_argument('--viz-torchviz', action='store_true', help='Visualize using torchviz')
    parser.add_argument('--viz-hiddenlayer', action='store_true', help='Visualize using hiddenlayer')
    parser.add_argument('--viz-netron', action='store_true', help='Visualize using netron')
    parser.add_argument('--viz-features', action='store_true', help='Visualize feature maps')
    parser.add_argument('--viz-all', action='store_true', help='Use all visualization methods')
    parser.add_argument('--compare', help='Path to second model file to compare with')
    parser.add_argument('--output-name', default='model', help='Base name for output files')
    
    args = parser.parse_args()
    
    # Get model code either from file or direct input
    model_code = None
    if args.model:
        model_code = read_model_from_file(args.model)
        if not model_code:
            print(f"Could not read model from file: {args.model}")
            return
    elif args.code:
        model_code = args.code
    
    if not model_code:
        print("No model code provided")
        return
    
    # Save the model code to a file
    model_file = save_generated_model(model_code, f"{args.output_name}.py")
    
    # Load the model
    model = load_model_from_file(model_file)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Parse input size
    input_size = parse_input_size(args.input_size)
    
    # Test forward pass
    print("\nTesting model forward pass:")
    test_model_forward(model, input_size)
    
    # Visualize the model using different methods
    if args.viz_all or args.viz_torchviz:
        print("\nGenerating torchviz visualization...")
        visualize_model_torchviz(model, input_size, f"{args.output_name}_torchviz")
    
    if args.viz_all or args.viz_hiddenlayer:
        print("\nGenerating hiddenlayer visualization...")
        visualize_model_hiddenlayer(model, input_size, f"{args.output_name}_hiddenlayer")
    
    if args.viz_all or args.viz_netron:
        print("\nGenerating netron visualization...")
        try:
            visualize_model_netron(model, input_size, f"{args.output_name}_netron.onnx")
        except Exception as e:
            print(f"Error with Netron visualization: {e}")
    
    if args.viz_all or args.viz_features:
        print("\nGenerating feature maps...")
        try:
            os.makedirs(f"{args.output_name}_features", exist_ok=True)
            trace_through_model(model, input_size, output_dir=f"{args.output_name}_features")
        except Exception as e:
            print(f"Error generating feature maps: {e}")
    
    # Compare models if requested
    if args.compare:
        print("\nComparing models...")
        compare_model_code = read_model_from_file(args.compare)
        if compare_model_code:
            compare_models(model_code, compare_model_code, input_size)
        else:
            print(f"Could not read comparison model from file: {args.compare}")
    
    print(f"\nAll outputs saved with base name: {args.output_name}")
    print("Done!")

if __name__ == "__main__":
    main() 