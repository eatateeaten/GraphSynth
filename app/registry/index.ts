import { validateParams as validateTorchParams } from '../../isomorphic/torch_nn_module_op';
import { NodeType, ModuleMetadata } from './types';
import { convolutionalModules } from './convolutional';
import { activationModules } from './activation';
import { normalizationModules } from './normalization';
import { linearModules } from './linear';
import { poolingModules } from './pooling';
import { dropoutModules } from './dropout';
import { reshapeModules } from './reshape';

// Re-export types

// Combine all module metadata
export const allModules: Record<string, ModuleMetadata> = {
  // Tensor nodes
  'Tensor': {
    label: 'Tensor',
    description: 'Represents input data or intermediate results',
    category: 'Basic',
    paramFields: {
      shape: {
        label: 'Shape',
        description: 'Dimensions of the tensor (comma-separated)',
        type: 'shape',
        default: [3, 64, 64],
        allowNegativeOne: false
      },
      isInput: {
        label: 'Input Node',
        description: 'Make this tensor an input node',
        type: 'boolean',
        default: false
      }
    }
  },
  // Operation nodes
  ...Object.entries(linearModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(convolutionalModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(reshapeModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(activationModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(dropoutModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(poolingModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  ...Object.entries(normalizationModules).reduce((acc, [key, value]) => ({ ...acc, [`Op:${key}`]: value }), {}),
  // Flow nodes
  'Merge': {
    label: 'Merge',
    description: 'Combines multiple inputs into a single output',
    category: 'Flow',
    paramFields: {}
  },
  'Branch': {
    label: 'Branch',
    description: 'Splits a single input into multiple outputs',
    category: 'Flow',
    paramFields: {}
  }
};

// Get metadata for any node type and operation
export function getMeta(nodeType: NodeType, opType?: string): ModuleMetadata {
  const key = opType ? `${nodeType}:${opType}` : nodeType;
  if (!allModules[key]) {
    throw new Error(`No metadata found for node type ${nodeType} and opType ${opType}`);
  }
  return allModules[key];
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
    "op": allModules
};
