import { ModuleMetadata, ParamFieldMetadata } from './types';

const commonDropoutParams: Record<string, ParamFieldMetadata> = {
  p: {
    label: 'Probability',
    description: 'Probability of an element to be zeroed',
    type: 'number',
    default: 0.5
  },
  inplace: {
    label: 'Inplace',
    description: 'If set to True, will do this operation in-place',
    type: 'option',
    options: ['true', 'false'],
    default: 'false'
  }
};

function createDropoutModule(dimension: '1D' | '2D' | '3D' | null): ModuleMetadata {
  const inputDesc = dimension ? `${dimension} input` : 'input';
  return {
    label: dimension ? `Dropout${dimension}` : 'Dropout',
    description: `Randomly zeroes some of the elements of the ${inputDesc} with probability p`,
    category: 'Dropout',
    paramFields: commonDropoutParams
  };
}

export const dropoutModules: Record<string, ModuleMetadata> = {
  'Dropout': createDropoutModule(null),
  'Dropout2D': createDropoutModule('2D'),
  'Dropout3D': createDropoutModule('3D'),
  'AlphaDropout': {
    label: 'Alpha Dropout',
    description: 'Applies Alpha Dropout over the input',
    category: 'Dropout',
    paramFields: commonDropoutParams
  },
  'Threshold': {
    label: 'Threshold',
    description: 'Thresholds each element of the input Tensor',
    category: 'Dropout',
    paramFields: {
      threshold: {
        label: 'Threshold',
        description: 'The value to threshold at',
        type: 'number',
        default: 0
      },
      value: {
        label: 'Value',
        description: 'The value to replace with',
        type: 'number',
        default: 0
      },
      inplace: commonDropoutParams.inplace
    }
  }
};
