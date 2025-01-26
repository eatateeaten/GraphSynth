import { useState, useCallback } from 'react';
import { Select, Button } from '@mantine/core';
import {
  ConvTypes,
  LinearTypes,
  FlattenTypes,
  ElementWiseNonlinearityTypes,
  Nonlinearity1DTypes,
  PoolTypes,
  AdaptivePoolTypes,
  formatLabel,
  type LayerType,
  type Layer,
  type LayerParams
} from './types';

const layerTypes = [
  ...ConvTypes,
  ...LinearTypes,
  ...FlattenTypes,
  ...ElementWiseNonlinearityTypes,
  ...Nonlinearity1DTypes,
  ...PoolTypes,
  ...AdaptivePoolTypes
].map(value => ({
  value,
  label: formatLabel(value)
}));

interface SidebarProps {
  onAddLayer: (layer: Layer) => void;
}

export function Sidebar({ onAddLayer }: SidebarProps) {
  const [value, setValue] = useState<string | null>(null);

  const handleAdd = useCallback(() => {
    if (!value) return;
    
    const type = value as LayerType;
    const layer = {
      id: crypto.randomUUID(),
      name: `${type}_${Date.now()}`,
      type,
      params: getDefaultParams(type),
      inputNodes: []
    };
    onAddLayer(layer);
    setValue(null);
  }, [value, onAddLayer]);

  return (
    <div style={{ padding: '16px' }}>
      <div style={{ display: 'flex', gap: '8px' }}>
        <Select
          placeholder="Choose Layer..."
          data={layerTypes}
          value={value}
          onChange={setValue}
          searchable
          clearable
          maxDropdownHeight={200}
        />
        <Button 
          onClick={handleAdd} 
          disabled={!value}
          variant="filled"
        >
          Add
        </Button>
      </div>
    </div>
  );
}

function getDefaultParams(type: LayerType): LayerParams {
  switch (type) {
    case 'conv1d':
      return {
        batch_size: 1,
        in_channels: 1,
        out_channels: 1,
        input_size: 32,
        kernel_size: 3,
        input_features: 0,
        output_features: 0
      };
    case 'linear':
      return {
        batch_size: 1,
        input_features: 10,
        output_features: 10,
        in_channels: 0,
        out_channels: 0,
        input_size: 0,
        kernel_size: 0
      };
    default:
      return {};
  }
} 