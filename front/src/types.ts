// Layer types
export const ConvTypes = ['conv1d', 'conv2d', 'conv3d'] as const;
export type ConvType = typeof ConvTypes[number];

export const LinearTypes = ['linear'] as const;
export type LinearType = typeof LinearTypes[number];

export const FlattenTypes = ['flatten'] as const;
export type FlattenType = typeof FlattenTypes[number];

export const ElementWiseNonlinearityTypes = [
  'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu', 'celu', 'gelu',
  'softplus', 'softsign', 'hardtanh', 'hardshrink', 'hardsigmoid', 'hardswish',
  'softshrink', 'tanhshrink', 'threshold', 'relu6', 'silu', 'mish'
] as const;
export type ElementWiseNonlinearityType = typeof ElementWiseNonlinearityTypes[number];

export const Nonlinearity1DTypes = ['softmax', 'log_softmax', 'glu'] as const;
export type Nonlinearity1DType = typeof Nonlinearity1DTypes[number];

export const PoolTypes = [
  'maxpool1d', 'maxpool2d', 'maxpool3d',
  'avgpool1d', 'avgpool2d', 'avgpool3d',
  'lppool1d', 'lppool2d', 'lppool3d'
] as const;
export type PoolType = typeof PoolTypes[number];

export const AdaptivePoolTypes = [
  'adaptive_maxpool1d', 'adaptive_maxpool2d', 'adaptive_maxpool3d',
  'adaptive_avgpool1d', 'adaptive_avgpool2d', 'adaptive_avgpool3d'
] as const;
export type AdaptivePoolType = typeof AdaptivePoolTypes[number];

export type LayerType = 
  | ConvType 
  | LinearType 
  | FlattenType 
  | ElementWiseNonlinearityType 
  | Nonlinearity1DType 
  | PoolType 
  | AdaptivePoolType;

export type LayerParams = {
  [key: string]: number | number[] | string;
};

export type Layer = {
  id: string;
  name: string;
  type: LayerType;
  params: LayerParams;
  inputNodes: string[];
};

export type Sequence = {
  name: string;
  nodes: Layer[];
};

export type WSResponse = {
  success: boolean;
  id?: string;
  pytorch_code?: string;
  error?: string;
};

// Helper to convert snake_case to Title Case
export function formatLabel(value: string): string {
  return value
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join('');
} 