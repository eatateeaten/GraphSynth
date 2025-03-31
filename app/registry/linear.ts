import { ModuleMetadata, ParamFieldMetadata } from './types';

const linearParams: Record<string, ParamFieldMetadata> = {
  input_features: {
    label: 'Input Features',
    description: 'Size of each input sample',
    type: 'number',
    default: 512
  },
  output_features: {
    label: 'Output Features',
    description: 'Size of each output sample',
    type: 'number',
    default: 256
  },
  bias: {
    label: 'Bias',
    description: 'Whether to add a learnable bias to the output',
    type: 'option',
    options: ['true', 'false'],
    default: 'true'
  }
};

export const linearModules: Record<string, ModuleMetadata> = {
  'Linear': {
    label: 'Linear',
    description: 'Applies a linear transformation to the incoming data',
    category: 'Linear',
    paramFields: linearParams
  },
  'Identity': {
    label: 'Identity',
    description: 'A placeholder identity operator that is argument-insensitive',
    category: 'Linear',
    paramFields: {}
  }
};
