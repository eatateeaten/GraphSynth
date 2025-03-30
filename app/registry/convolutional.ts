import { ModuleMetadata, ParamFieldMetadata } from './types';

// Base parameter templates for common parameters
const commonParams: Record<string, ParamFieldMetadata> = {
  in_channels: {
    label: 'Input Channels',
    description: 'Number of channels in the input',
    type: 'number',
    default: 3
  },
  out_channels: {
    label: 'Output Channels',
    description: 'Number of channels produced by the convolution',
    type: 'number',
    default: 16
  },
  kernel_size: {
    label: 'Kernel Size',
    description: 'Size of the convolving kernel',
    type: 'number',
    default: 3
  },
  stride: {
    label: 'Stride',
    description: 'Stride of the convolution',
    type: 'number',
    default: 1
  },
  padding: {
    label: 'Padding',
    description: 'Padding added to the input',
    type: 'number',
    default: 0
  },
  dilation: {
    label: 'Dilation',
    description: 'Spacing between kernel elements',
    type: 'number',
    default: 1
  },
  groups: {
    label: 'Groups',
    description: 'Number of blocked connections from input to output channels',
    type: 'number',
    default: 1
  },
  bias: {
    label: 'Bias',
    description: 'Whether to add a learnable bias to the output',
    type: 'option',
    options: ['true', 'false'],
    default: 'true'
  },
  padding_mode: {
    label: 'Padding Mode',
    description: 'Type of padding algorithm',
    type: 'option',
    options: ['zeros', 'reflect', 'replicate', 'circular'],
    default: 'zeros'
  }
};

// Additional parameters for transposed convolutions
const transposeParams: Record<string, ParamFieldMetadata> = {
  output_padding: {
    label: 'Output Padding',
    description: 'Additional size added to one side of each dimension in the output shape',
    type: 'number',
    default: 0
  }
};

// Factory function for conventional convolutions
function createConvModule(dimension: '1D' | '2D' | '3D'): ModuleMetadata {
  // Customize descriptions based on dimension
  const inputDesc = dimension === '1D' ? 'signal' : 
                   dimension === '2D' ? 'image' : 'volume';
  
  return {
    label: `Convolution ${dimension}`,
    description: `Applies a ${dimension} convolution over an input ${inputDesc}`,
    category: 'Convolutional',
    paramFields: {
      ...commonParams,
      in_channels: {
        ...commonParams.in_channels,
        description: `Number of channels in the input ${inputDesc}`
      },
      padding: {
        ...commonParams.padding,
        description: dimension === '1D' 
          ? 'Padding added to both sides of the input'
          : 'Padding added to all sides of the input'
      }
    }
  };
}

// Factory function for transposed convolutions
function createTransposeConvModule(dimension: '1D' | '2D' | '3D'): ModuleMetadata {
  // Customize descriptions based on dimension
  const inputDesc = dimension === '1D' ? 'signal' : 
                   dimension === '2D' ? 'image' : 'volume';
  
  // For transposed convolutions, default to more outputs than inputs
  return {
    label: `Transposed Convolution ${dimension}`,
    description: `Applies a ${dimension} transposed convolution operator over an input ${inputDesc}`,
    category: 'Convolutional',
    paramFields: {
      ...commonParams,
      ...transposeParams,
      in_channels: {
        ...commonParams.in_channels,
        description: `Number of channels in the input ${inputDesc}`,
        default: 16 // Typically more input channels for transposed
      },
      out_channels: {
        ...commonParams.out_channels,
        description: `Number of channels produced by the convolution`,
        default: 3  // Typically fewer output channels for transposed
      },
      padding: {
        ...commonParams.padding,
        description: dimension === '1D' 
          ? 'Padding added to both sides of the input'
          : 'Padding added to all sides of the input'
      }
    }
  };
}

// Generate all convolutional module definitions
export const convolutionalModules: Record<string, ModuleMetadata> = {
  // Regular convolutions
  'Conv1D': createConvModule('1D'),
  'Conv2D': createConvModule('2D'),
  'Conv3D': createConvModule('3D'),
  
  // Transposed convolutions
  'ConvTranspose1D': createTransposeConvModule('1D'),
  'ConvTranspose2D': createTransposeConvModule('2D'),
  'ConvTranspose3D': createTransposeConvModule('3D')
};
