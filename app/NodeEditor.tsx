import { useState, useCallback, useEffect } from 'react';
import { Select, Button, TextInput, Box, Text, Checkbox } from '@mantine/core';
import { useStore } from './store';
import { NodeType, ParamFieldMetadata, Shape } from '../registry/types';
import { allModules, getMeta, validateParams } from '../registry/index';

/* Build layer type options from available modules */
const LAYER_TYPE_OPTIONS = (() => {
    // Create lists of all operation types
    const opTypes = Object.keys(allModules)
        .filter(key => key.startsWith('Op:'))
        .map(key => ({ key, label: allModules[key].label }));
    
    // Get direct node types
    const directTypes = ['Tensor', 'Split', 'Copy', 'Concat', 'PointwiseReduce', 'DotOp', 'CrossOp']
        .filter(key => allModules[key])
        .map(key => ({ key, label: allModules[key].label }));

    let out = [{ 
        group: 'Tensor', 
        items: [{ value: 'Tensor', label: 'Tensor' }] 
    }];

    // Add Op types
    out.push(...opTypes.reduce((groups, { key, label }) => {
        const metadata = getMeta(key);
        const category = metadata.category;
        const item = { value: key, label };
        
        const existingGroup = groups.find(g => g.group === category);
        if (existingGroup) {
            existingGroup.items.push(item);
        } else {
            groups.push({ group: category, items: [item] });
        }

        return groups;
    }, [] as Array<{ group: string; items: Array<{ value: string; label: string }>}>));

    // Add direct node types (excluding Tensor which is already added)
    out.push(...directTypes
        .filter(({ key }) => key !== 'Tensor')
        .reduce((groups, { key, label }) => {
            const metadata = getMeta(key);
            const category = metadata.category;
            const item = { value: key, label };
            
            const existingGroup = groups.find(g => g.group === category);
            if (existingGroup) {
                existingGroup.items.push(item);
            } else {
                groups.push({ group: category, items: [item] });
            }

            return groups;
        }, [] as Array<{ group: string; items: Array<{ value: string; label: string }>}>));

    return out;
})();

export function NodeEditor() {
    const [moduleKey, setModuleKey] = useState<string | null>(null);
    const [params, setParams] = useState<Record<string, any>>({});
    const [rawInputs, setRawInputs] = useState<Record<string, string>>({});
    const [touchedFields, setTouchedFields] = useState<Set<string>>(new Set());
    const [error, setError] = useState<string | null>(null);

    const addNode = useStore(state => state.addNode);
    const selectedId = useStore(state => state.selectedId);
    const updateNodeParams = useStore(state => state.updateNodeParams);
    const nodes = useStore(state => state.nodes);
    const selectedNodeData = selectedId ? nodes.find(n => n.id === selectedId) : null;

    // Get default params for a module
    const getDefaultParams = useCallback((moduleKey: string) => {
        const metadata = getMeta(moduleKey);
        const defaults: Record<string, any> = {};
        
        Object.entries(metadata.paramFields).forEach(([name, field]) => {
            if (field.default !== undefined) {
                defaults[name] = field.default;
            }
        });

        return defaults;
    }, []);

    // Initialize raw inputs from params
    const paramsToRawInputs = useCallback((params: Record<string, any>) => {
        const rawValues: Record<string, string> = {};
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null) {
                rawValues[key] = value.toString();
            }
        });
        return rawValues;
    }, []);

    // Reset raw inputs when type changes
    const handleTypeChange = (v: string | null) => {
        setModuleKey(v);
        if (v) {
            const defaults = getDefaultParams(v);
            setParams(defaults);
            setRawInputs(paramsToRawInputs(defaults));
        } else {
            setParams({});
            setRawInputs({});
        }
        setTouchedFields(new Set());
        setError(null);
    };

    const validateModuleParams = useCallback((params: Record<string, any>, moduleKey: string): string | null => {
        // If it's not an Op, no validation needed
        if (!moduleKey.startsWith('Op:')) {
            return null;
        }
        
        // For ops, validate using the op type
        const opType = moduleKey.slice(3);
        return validateParams(opType, params);
    }, []);

    const handleParamChange = useCallback((name: string, newValue: any) => {
        setTouchedFields(fields => new Set([...fields, name]));
        const newParams = { ...params, [name]: newValue };
        setParams(newParams);
        
        if (selectedId) {
            updateNodeParams(selectedId, newParams);
        }
    }, [selectedId, params, updateNodeParams]);

    const parseDimensions = useCallback((input: string, allowNegativeOne = false): Shape | null => {
        try {
            const dims = input.split(',').map(num => {
                const parsed = parseInt(num.trim(), 10);
                if (isNaN(parsed)) throw new Error('Invalid number');
                if (!allowNegativeOne && parsed <= 0) throw new Error('Must be positive');
                if (allowNegativeOne && parsed !== -1 && parsed <= 0) throw new Error('Must be positive or -1');
                return parsed;
            });
            return dims;
        } catch {
            return null;
        }
    }, []);

    const renderParamField = (name: string, field: ParamFieldMetadata) => {
        const value = params[name];
        const rawValue = rawInputs[name] ?? '';
        const showError = touchedFields.has(name);
        const validationError = showError && moduleKey ? validateModuleParams(params, moduleKey) : undefined;

        switch (field.type) {
        case 'boolean':
            return (
                <Checkbox
                    key={name}
                    label={field.label}
                    description={field.description}
                    checked={value || false}
                    onChange={(e) => handleParamChange(name, e.currentTarget.checked)}
                />
            );
        case 'shape':
            return (
                <TextInput
                    key={name}
                    label={field.label}
                    description={field.description}
                    placeholder={field.allowNegativeOne ? '3,4,-1' : '3,8,3'}
                    value={rawValue}
                    onChange={(e) => {
                        const val = e.target.value;
                        setRawInputs(p => ({ ...p, [name]: val }));
                            
                        // Try to parse but don't prevent invalid input
                        const shape = parseDimensions(val, field.allowNegativeOne);
                        handleParamChange(name, shape || undefined);
                    }}
                    error={validationError}
                    onFocus={() => setTouchedFields(fields => new Set([...fields, name]))}
                />
            );
        case 'number':
            return (
                <TextInput
                    key={name}
                    label={field.label}
                    description={field.description}
                    placeholder="1"
                    value={typeof value === 'number' ? value.toString() : ''}
                    onChange={(e) => {
                        const num = parseInt(e.target.value, 10);
                        handleParamChange(name, !isNaN(num) && num > 0 ? num : undefined);
                    }}
                    error={validationError}
                    onFocus={() => setTouchedFields(fields => new Set([...fields, name]))}
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
                    onChange={(val) => handleParamChange(name, val)}
                    error={validationError}
                    onFocus={() => setTouchedFields(fields => new Set([...fields, name]))}
                />
            );
        case 'string':
            return (
                <TextInput
                    key={name}
                    label={field.label}
                    description={field.description}
                    placeholder={field.placeholder || ""}
                    value={value || ""}
                    onChange={(e) => handleParamChange(name, e.target.value)}
                    error={validationError}
                    onFocus={() => setTouchedFields(fields => new Set([...fields, name]))}
                />
            );
        }
    };

    const handleAdd = useCallback(() => {
        if (!moduleKey) return;
        
        try {
            const id = crypto.randomUUID();
            
            // Handle different node types based on the module key
            if (moduleKey === 'Tensor') {
                const shape = params.shape;
                if (!shape || !Array.isArray(shape)) {
                    setError('A valid shape is required for tensor nodes');
                    return;
                }
                
                addNode({
                    id,
                    type: 'Tensor',
                    params
                });
            } else if (moduleKey.startsWith('Op:')) {
                // For Op types
                const opType = moduleKey.slice(3);
                
                // Validate parameters
                const validationError = validateModuleParams(params, moduleKey);
                if (validationError) {
                    setError(validationError);
                    return;
                }

                addNode({
                    id,
                    type: 'Op',
                    params: {...params, opType}
                });
            } else {
                // For direct node types (Split, Copy, Concat, PointwiseReduce)
                addNode({
                    id,
                    type: moduleKey as NodeType,
                    params
                });
            }
            
            setModuleKey(null);
            setParams({});
            setRawInputs({});
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to create node');
        }
    }, [moduleKey, params, addNode, validateModuleParams]);

    // Get metadata for the selected module key
    const metadata = moduleKey ? getMeta(moduleKey) : null;

    useEffect(() => {
        if (selectedId && selectedNodeData) {
            // Get data from the store's nodes array
            const type = selectedNodeData.data.type;
            const opType = selectedNodeData.data.opType;
            
            let newModuleKey: string | null = null;
            
            // Convert node type + opType into the corresponding module key
            if (type === 'Op' && opType) {
                newModuleKey = `Op:${opType}`;
            } else if (['Tensor', 'Split', 'Copy', 'Concat', 'PointwiseReduce', 'DotOp', 'CrossOp'].includes(type)) {
                newModuleKey = type;
            }
            
            setModuleKey(newModuleKey);
            setParams(selectedNodeData.data.params || {});
            setRawInputs(paramsToRawInputs(selectedNodeData.data.params || {}));
            setError(null);
        } else if (!selectedId) {
            setModuleKey(null);
            setParams({});
            setRawInputs({});
            setError(null);
        }
    }, [selectedId, selectedNodeData, paramsToRawInputs]);

    return (
        <Box p="md" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <Select
                label="Layer Type"
                placeholder="Choose Layer..."
                data={LAYER_TYPE_OPTIONS}
                value={moduleKey}
                onChange={handleTypeChange}
                searchable
                clearable
                disabled={!!selectedId}
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

            {error && (
                <Text color="red" size="sm">
                    {error}
                </Text>
            )}

            {!selectedId && (
                <Button 
                    onClick={handleAdd} 
                    disabled={!moduleKey || !metadata || validateModuleParams(params, moduleKey) !== null}
                    variant="filled"
                >
                    Add Layer
                </Button>
            )}
        </Box>
    );
}
