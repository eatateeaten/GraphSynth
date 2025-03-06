import { useState, useCallback } from 'react';
import { Select, Button, TextInput, Box, Text, NumberInput } from '@mantine/core';
import type { CheckerNodeType, CheckerNodeConfig, CheckerNodeParams } from './checker';
import { useGraphStore } from './store';
import { ParamFieldMetadata, NodeMetadata } from './checker/node';
import { Shape } from './checker/shape';

export function Sidebar() {
  const [type, setType] = useState<CheckerNodeType | null>(null);
  const [params, setParams] = useState<Partial<CheckerNodeParams>>({});
  const addNode = useGraphStore(state => state.addNode);

  const handleAdd = useCallback(() => {
    if (!type || !params) return;
    
    const selectedNode = nodeTypes.find(n => n.type === type);
    if (!selectedNode) return;

    const metadata = selectedNode.getMeta();
    const isValid = validateParams(params, metadata);
    if (!isValid) return;

    const config: CheckerNodeConfig = {
      type,
      params: params as CheckerNodeParams[typeof type]
    };

    const id = crypto.randomUUID();
    addNode(id, config);
    setType(null);
    setParams({});
  }, [type, params, addNode]);

  const parseDimensions = useCallback((input: string, allowNegativeOne = false): Shape | null => {
    try {
      const dims = input.split(',').map(num => {
        const parsed = parseInt(num.trim(), 10);
        if (isNaN(parsed)) throw new Error('Invalid number');
        if (!allowNegativeOne && parsed <= 0) throw new Error('Must be positive');
        if (allowNegativeOne && parsed !== -1 && parsed <= 0) throw new Error('Must be positive or -1');
        return parsed;
      });
      return new Shape(dims, allowNegativeOne);
    } catch (e) {
      return null;
    }
  }, []);

  const validateParams = (params: Partial<NodeParams>, metadata: NodeMetadata): params is NodeParams => {
    return Object.entries(metadata.paramFields).every(([name, field]) => {
      const value = params[name];
      if (value === undefined) return false;

      switch (field.type) {
        case 'shape':
          return value instanceof Shape && value.length > 0;
        case 'number':
          return typeof value === 'number' && value > 0;
        case 'option':
          return typeof value === 'string' && field.options?.includes(value);
        default:
          return false;
      }
    });
  };

  const renderParamField = (name: string, field: ParamFieldMetadata) => {
    const value = params[name];

    switch (field.type) {
      case 'shape':
        return (
          <TextInput
            key={name}
            label={field.label}
            description={field.description}
            placeholder={field.allowNegativeOne ? '3,4,-1' : '3,8,3'}
            value={value instanceof Shape ? value.join(',') : ''}
            onChange={(e) => {
              const shape = parseDimensions(e.target.value, field.allowNegativeOne);
              if (shape) setParams(p => ({ ...p, [name]: shape }));
            }}
            error={value === undefined ? 'Required' : undefined}
          />
        );
      case 'number':
        return (
          <NumberInput
            key={name}
            label={field.label}
            description={field.description}
            min={1}
            value={typeof value === 'number' ? value : undefined}
            onChange={(val) => setParams(p => ({ ...p, [name]: val }))}
            error={value === undefined ? 'Required' : undefined}
          />
        );
      case 'option':
        return (
          <Select
            key={name}
            label={field.label}
            description={field.description}
            data={field.options || []}
            value={typeof value === 'string' ? value : null}
            onChange={(val) => setParams(p => ({ ...p, [name]: val }))}
            error={value === undefined ? 'Required' : undefined}
          />
        );
    }
  };

  const selectedNode = type && nodeTypes.find(n => n.type === type);
  const metadata = selectedNode?.getMeta();

  return (
    <Box p="md" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <Select
        label="Layer Type"
        placeholder="Choose Layer..."
        data={nodeTypes.map(node => ({
          value: node.type,
          label: node.getMeta().label,
          group: node.getMeta().category
        }))}
        value={type}
        onChange={(v) => {
          setType(v as NodeType);
          setParams({});
        }}
        searchable
        clearable
      />
      
      {metadata && (
        <Box style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <Text size="sm" color="dimmed">
            {metadata.description}
          </Text>
          {Object.entries(metadata.paramFields).map(([name, field]) => 
            renderParamField(name, field)
          )}
        </Box>
      )}

      <Button 
        onClick={handleAdd} 
        disabled={!type || !metadata || !validateParams(params, metadata)}
        variant="filled"
      >
        Add Layer
      </Button>
    </Box>
  );
}
