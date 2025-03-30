import { validateParams as validateTorchParams } from '../../isomorphic/torch_nn_module_op';
import { NodeType, ModuleMetadata } from './types';
import { convolutionalModules } from './convolutional';
import { activationModules } from './activation';
import { normalizationModules } from './normalization';

// Re-export types

// Combine all module metadata
const allModules: Record<string, ModuleMetadata> = {
  ...convolutionalModules,
  ...activationModules,
  ...normalizationModules,
  // Add more module categories as they're created
};

// Module for tensor nodes
const tensorMetadata: ModuleMetadata = {
  label: 'Tensor',
  description: 'A tensor node represents input data or intermediate results',
  category: 'Basic',
  paramFields: {
    shape: {
      label: 'Shape',
      description: 'Dimensions of the tensor (comma-separated)',
      type: 'shape',
      allowNegativeOne: false
    }
  }
};

// Get metadata for any node type and operation
export function getMeta(nodeType: NodeType, opType?: string): ModuleMetadata {
  if (nodeType === 'tensor') {
    return tensorMetadata;
  }
  
  if (nodeType === 'op' && opType && allModules[opType]) {
    return allModules[opType];
  }
  
  if (nodeType === 'merge') {
    return {
      label: 'Merge',
      description: 'Combines multiple inputs into a single output',
      category: 'Flow',
      paramFields: {
        // Will be expanded when implementing merge nodes
      }
    };
  }
  
  if (nodeType === 'branch') {
    return {
      label: 'Branch',
      description: 'Splits a single input into multiple outputs',
      category: 'Flow',
      paramFields: {
        // Will be expanded when implementing branch nodes
      }
    };
  }
  
  throw new Error(`No metadata found for node type ${nodeType} and opType ${opType}`);
}

// Validate parameters for an operation
export function validateParams(opType: string, params: Record<string, any>): string | null {
  try {
    validateTorchParams(opType, params);
    return null;
  } catch (e: any) {
    return e.message;
  }
}

/* category -> opType -> metadata */
export const ModuleRegistry = {
    "op": allModules,
    "tensor": tensorMetadata
};
