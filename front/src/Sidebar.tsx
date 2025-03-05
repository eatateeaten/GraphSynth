import { useState, useCallback } from 'react';
import { Select, Button, TextInput, Box, Text } from '@mantine/core';
import type { CheckerNodeType } from './checker';
import { useGraphStore } from './store';

const layerTypes = [
  { value: 'tensor', label: 'Tensor' },
  { value: 'reshape', label: 'Reshape' }
] as const;

export function Sidebar() {
  const [type, setType] = useState<CheckerNodeType | null>(null);
  const [params, setParams] = useState<Record<string, any>>({});
  const addNode = useGraphStore(state => state.addNode);

  const handleAdd = useCallback(() => {
    if (!type) return;
    
    const config = {
      type,
      params,
    };

    const id = crypto.randomUUID();
    addNode(id, config);
    setType(null);
    setParams({});
  }, [type, params, addNode]);

  const parseDimensions = (input: string, allowNegativeOne = false): number[] | null => {
    try {
      return input.split(',').map(num => {
        const parsed = parseInt(num.trim(), 10);
        if (isNaN(parsed)) throw new Error('Invalid number');
        if (!allowNegativeOne && parsed <= 0) throw new Error('Must be positive');
        if (allowNegativeOne && parsed !== -1 && parsed <= 0) throw new Error('Must be positive or -1');
        return parsed;
      });
    } catch (e) {
      return null;
    }
  };

  const renderParamsInput = () => {
    if (!type) return null;

    switch (type) {
      case 'tensor':
        return (
          <Box style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <TextInput
              label="Shape"
              description="Enter shape dimensions separated by commas (e.g., 3,8,3)"
              placeholder="3,8,3"
              onChange={(e) => {
                const shape = parseDimensions(e.target.value);
                if (shape) setParams({ shape });
              }}
            />
            <Text size="sm" color="dimmed">
              Creates a tensor with the specified shape.
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
                const out_dim = parseDimensions(e.target.value, true);
                if (out_dim) setParams({ out_dim });
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
    if (!type || !params) return false;
    
    switch (type) {
      case 'tensor':
        return Array.isArray(params.shape) && params.shape.length > 0;
      case 'reshape':
        return Array.isArray(params.out_dim) && params.out_dim.length > 0;
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
        value={type}
        onChange={(v) => {
          setType(v as CheckerNodeType);
          setParams({});
        }}
        searchable
        clearable
      />
      {renderParamsInput()}
      <Button 
        onClick={handleAdd} 
        disabled={!type || !isValidParams()}
        variant="filled"
      >
        Add Layer
      </Button>
    </Box>
  );
}
