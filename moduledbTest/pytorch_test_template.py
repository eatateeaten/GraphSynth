#!/usr/bin/env python3
"""
Simple PyTorch Module Test Template
This template:
1. Compiles generated PyTorch module code
2. Creates randomized input tensors 
3. Executes forward pass
4. Returns results for further analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import traceback
from typing import List, Dict, Any


def test_pytorch_module(module_code: str, input_shapes: List[List[int]]) -> Dict[str, Any]:
    """
    Test a generated PyTorch module with randomized inputs.
    
    Args:
        module_code: Generated PyTorch module code as string
        input_shapes: List of input tensor shapes [[batch, ...], [batch, ...]]
    
    Returns:
        Dictionary with test results
    """
    result = {
        "compilation_success": False,
        "forward_success": False,
        "module_code": module_code,
        "input_shapes": input_shapes,
        "output_shapes": None,
        "output_tensors": None,
        "error_message": None
    }
    
    try:
        # Step 1: Compile the module code
        local_namespace = {
            'torch': torch,
            'nn': nn,
            'F': F
        }
        exec(module_code, local_namespace)
        
        if 'GeneratedModule' not in local_namespace:
            result["error_message"] = "GeneratedModule class not found in compiled code"
            return result
        
        result["compilation_success"] = True
        module_class = local_namespace['GeneratedModule']
        
        # Step 2: Create module instance
        module = module_class()
        module.eval()  # Set to evaluation mode
        
        # Step 3: Generate randomized input tensors
        input_tensors = [torch.randn(*shape) for shape in input_shapes]
        
        # Step 4: Execute forward pass
        if len(input_tensors) == 1:
            # Single input
            output = module(input_tensors[0])
        else:
            # Multiple inputs
            output = module(*input_tensors)
        
        # Step 5: Process outputs
        if isinstance(output, torch.Tensor):
            output_shapes = [list(output.shape)]
            output_tensors = [output.detach().numpy().tolist()]
        elif isinstance(output, (list, tuple)):
            output_shapes = [list(o.shape) for o in output]
            output_tensors = [o.detach().numpy().tolist() for o in output]
        else:
            result["error_message"] = f"Unexpected output type: {type(output)}"
            return result
        
        result["forward_success"] = True
        result["output_shapes"] = output_shapes
        result["output_tensors"] = output_tensors
        
    except Exception as e:
        result["error_message"] = f"Error: {str(e)}\n{traceback.format_exc()}"
    
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pytorch_test_template.py <module_code_string> <input_shapes_json>")
        print("Example: python pytorch_test_template.py 'import torch...' '[[32, 784], [32, 128]]'")
        sys.exit(1)
    
    module_code = sys.argv[1]
    input_shapes_json = sys.argv[2]
    
    try:
        input_shapes = json.loads(input_shapes_json)
        
        results = test_pytorch_module(module_code, input_shapes)
        
        # Output results as JSON for consumption by TypeScript tests
        print(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if results["compilation_success"] and results["forward_success"] else 1)
        
    except Exception as e:
        error_result = {
            "compilation_success": False,
            "forward_success": False,
            "error_message": f"Script error: {str(e)}\n{traceback.format_exc()}"
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1) 