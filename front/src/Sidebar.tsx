import { useState, useCallback } from 'react';
import { Select, Button, TextInput, Box, Text } from '@mantine/core';
import type { LayerType, LayerParams, Sourceness } from './types';
import { useGraphStore } from './store';

const layerTypes = [
  { value: 'tensor', label: 'Tensor' },
  { value: 'reshape', label: 'Reshape' }
];

export function Sidebar() {
  const [value, setValue] = useState<string | null>(null);
  const [params, setParams] = useState<LayerParams>({});
  const addLayer = useGraphStore(state => state.addLayer);

  const handleAdd = useCallback(() => {
    if (!value) return;
    
    const type = value as LayerType;
    // Determine whether node generates, transforms, or consumes data
    const sourceness: Sourceness = type === 'tensor' ? 'source' : 'middle';
    
    const layer = {
      id: crypto.randomUUID(),
      name: `${type}_${Date.now()}`,
      type,
      sourceness,
      params,
    };
    addLayer(layer);
    setValue(null);
    setParams({});
  }, [value, params, addLayer]);

  const parseTensorShape = (input: string): number[] | null => {
    try {
      // Parse input like "3,8,3" into [3,8,3]
      return input.split(',').map(num => {
        const parsed = parseInt(num.trim(), 10);
        if (isNaN(parsed) || parsed <= 0) throw new Error('Invalid dimension');
        return parsed;
      });
    } catch (e) {
      return null;
    }
  };

  const parseReshapeDims = (input: string): number[] | null => {
    try {
      // Parse input like "3,4,-1" into [3,4,-1]
      return input.split(',').map(num => {
        const parsed = parseInt(num.trim(), 10);
        if (isNaN(parsed) || (parsed <= 0 && parsed !== -1)) throw new Error('Invalid dimension');
        return parsed;
      });
    } catch (e) {
      return null;
    }
  };

  const renderParamsInput = () => {
    if (!value) return null;

    switch (value) {
      case 'tensor':
        return (
          <Box style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <TextInput
              label="Tensor Shape"
              description="Enter shape dimensions separated by commas (e.g., 3,8,3)"
              placeholder="3,8,3"
              onChange={(e) => {
                const shape = parseTensorShape(e.target.value);
                if (shape) {
                  // For tensors, we set both data and outShape directly
                  // Tensors don't have input shape and their output shape is fixed
                  const size = shape.reduce((a, b) => a * b, 1);
                  setParams({ 
                    data: new Array(size).fill(0),
                    shape: shape // Store the shape directly
                  });
                }
              }}
            />
            <Text size="sm" color="dimmed">
              Creates a tensor with the specified shape. Tensors are source nodes with a fixed shape.
            </Text>
          </Box>
        );
      case 'reshape':
        return (
          <Box style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <TextInput
              label="Output Dimensions"
              description="Enter dimensions separated by commas. Use -1 for automatic inference (e.g., 3,4,-1)"
              placeholder="3,4,-1"
              onChange={(e) => {
                const dims = parseReshapeDims(e.target.value);
                if (dims) {
                  setParams({ out_dim: dims });
                }
              }}
            />
            <Text size="sm" color="dimmed">
              At most one dimension can be -1 (will be inferred from other dimensions)
            </Text>
          </Box>
        );
      default:
        return null;
    }
  };

  const isValidParams = () => {
    if (!value) return false;
    
    switch (value) {
      case 'tensor':
        return params.data && params.data.length > 0;
      case 'reshape':
        return params.out_dim && params.out_dim.length > 0 && 
               params.out_dim.filter(x => x === -1).length <= 1;
      default:
        return false;
    }
  };

  return (
    <Box p="md" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <Select
        label="Layer Type"
        placeholder="Choose Layer..."
        data={layerTypes}
        value={value}
        onChange={(v) => {
          setValue(v);
          setParams({});
        }}
        searchable
        clearable
      />
      {renderParamsInput()}
      <Button 
        onClick={handleAdd} 
        disabled={!value || !isValidParams()}
        variant="filled"
      >
        Add Layer
      </Button>
    </Box>
  );
} 