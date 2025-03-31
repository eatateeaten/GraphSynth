import { ModuleMetadata, ParamFieldMetadata } from './types';

const commonPoolParams: Record<string, ParamFieldMetadata> = {
  kernel_size: {
    label: 'Kernel Size',
    description: 'Size of the pooling window',
    type: 'number',
    default: 2
  },
  stride: {
    label: 'Stride',
    description: 'Stride of the pooling operation',
    type: 'number',
    default: 2
  },
  ceil_mode: {
    label: 'Ceil Mode',
    description: 'When True, will use ceil instead of floor to compute the output shape',
    type: 'option',
    options: ['true', 'false'],
    default: 'false'
  }
};

const adaptivePoolParams: Record<string, ParamFieldMetadata> = {
  output_size: {
    label: 'Output Size',
    description: 'Target output size of the image',
    type: 'number',
    default: 1
  }
};

const lpPoolParams: Record<string, ParamFieldMetadata> = {
  ...commonPoolParams,
  norm_type: {
    label: 'Norm Type',
    description: 'Type of norm to use',
    type: 'number',
    default: 2
  }
};

function createPoolModule(dimension: '1D' | '2D' | '3D', type: 'Max' | 'Avg' | 'LP' | 'AdaptiveMax' | 'AdaptiveAvg'): ModuleMetadata {
  const inputDesc = dimension === '1D' ? 'signal' : dimension === '2D' ? 'image' : 'volume';
  const params = type === 'AdaptiveMax' || type === 'AdaptiveAvg' 
    ? adaptivePoolParams 
    : type === 'LP' 
      ? lpPoolParams 
      : commonPoolParams;

  return {
    label: `${type}Pool${dimension}`,
    description: `Applies ${type.toLowerCase()} pooling over a ${dimension} ${inputDesc}`,
    category: 'Pooling',
    paramFields: params
  };
}

export const poolingModules: Record<string, ModuleMetadata> = {
  // Regular pooling
  'MaxPool1D': createPoolModule('1D', 'Max'),
  'MaxPool2D': createPoolModule('2D', 'Max'),
  'MaxPool3D': createPoolModule('3D', 'Max'),
  'AvgPool1D': createPoolModule('1D', 'Avg'),
  'AvgPool2D': createPoolModule('2D', 'Avg'),
  'AvgPool3D': createPoolModule('3D', 'Avg'),
  'LPPool1D': createPoolModule('1D', 'LP'),
  'LPPool2D': createPoolModule('2D', 'LP'),
  
  // Adaptive pooling
  'AdaptiveMaxPool1D': createPoolModule('1D', 'AdaptiveMax'),
  'AdaptiveMaxPool2D': createPoolModule('2D', 'AdaptiveMax'),
  'AdaptiveMaxPool3D': createPoolModule('3D', 'AdaptiveMax'),
  'AdaptiveAvgPool1D': createPoolModule('1D', 'AdaptiveAvg'),
  'AdaptiveAvgPool2D': createPoolModule('2D', 'AdaptiveAvg'),
  'AdaptiveAvgPool3D': createPoolModule('3D', 'AdaptiveAvg')
};
