import { useState, useCallback, useEffect } from 'react';
import { Select, Button, TextInput, Box, Text, Checkbox } from '@mantine/core';
import { useStore } from './store';
import { NodeType, ParamFieldMetadata, Shape } from './registry/types';
import { allModules, getMeta, validateParams } from './registry/index';

/* Build layer type options from available modules */
const LAYER_TYPE_OPTIONS = (() => {
    // Create lists of all operation types
    const opTypes = Object.keys(allModules)
        .filter(key => key.startsWith('Op:'))
        .map(key => key.slice(3));
    
    const branchTypes = Object.keys(allModules)
        .filter(key => key.startsWith('Branch:'))
        .map(key => key.slice(7));
    
    const mergeTypes = Object.keys(allModules)
        .filter(key => key.startsWith('Merge:'))
        .map(key => key.slice(6));

    let out = [{ group: 'Tensor', items: [{ value: 'Tensor', label: 'Tensor' }] }];

    // Add Op types
    out.push(...opTypes.reduce((groups, type) => {
        const metadata = getMeta('Op', type);
        const category = metadata.category;
        const item = { value: type, label: metadata.label };
        
        const existingGroup = groups.find(g => g.group === category);
        if (existingGroup) {
            existingGroup.items.push(item);
        } else {
            groups.push({ group: category, items: [item] });
        }

        return groups;
    }, [] as Array<{ group: string; items: Array<{ value: string; label: string }>}>));

    // Add Branch types
    out.push(...branchTypes.reduce((groups, type) => {
        const metadata = getMeta('Branch', type);
        const item = { value: type, label: metadata.label };
        
        const existingGroup = groups.find(g => g.group === 'Flow');
        if (existingGroup) {
            existingGroup.items.push(item);
        } else {
            groups.push({ group: 'Flow', items: [item] });
        }

        return groups;
    }, [] as Array<{ group: string; items: Array<{ value: string; label: string }>}>));

    // Add Merge types
    out.push(...mergeTypes.reduce((groups, type) => {
        const metadata = getMeta('Merge', type);
        const item = { value: type, label: metadata.label };
        
        const existingGroup = groups.find(g => g.group === 'Flow');
        if (existingGroup) {
            existingGroup.items.push(item);
        } else {
            groups.push({ group: 'Flow', items: [item] });
        }

        return groups;
    }, [] as Array<{ group: string; items: Array<{ value: string; label: string }>}>));

    return out;
})();

export function Sidebar() {
    const [opType, setOpType] = useState<string | null>(null);
    const [params, setParams] = useState<Record<string, any>>({});
    const [rawInputs, setRawInputs] = useState<Record<string, string>>({});
    const [touchedFields, setTouchedFields] = useState<Set<string>>(new Set());
    const [error, setError] = useState<string | null>(null);

    const addNode = useStore(state => state.addNode);
    const selectedId = useStore(state => state.selectedId);
    const updateNodeParams = useStore(state => state.updateNodeParams);
    const nodes = useStore(state => state.nodes);
    const selectedNodeData = selectedId ? nodes.find(n => n.id === selectedId) : null;
    const makeTensorSource = useStore(state => state.makeTensorSource);

    // Get default params for a node type
    const getDefaultParams = useCallback((opType: string) => {
        const metadata = getMeta(opType === 'Tensor' ? 'Tensor' : 'Op', opType === 'Tensor' ? undefined : opType);
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
        setOpType(v);
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

    const validateModuleParams = useCallback((params: Record<string, any>, opType: string): string | null => {
        if(opType === 'Tensor')
            return null;
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
        } catch (e) {
            return null;
        }
    }, []);

    const renderParamField = (name: string, field: ParamFieldMetadata) => {
        const value = params[name];
        const rawValue = rawInputs[name] ?? '';
        const showError = touchedFields.has(name);
        const validationError = showError && opType ? validateModuleParams(params, opType) : undefined;

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
        }
    };

    const handleAdd = useCallback(() => {
        if (!opType) return;
        
        // Special handling for Tensor nodes
        if (opType === 'Tensor') {
            const shape = params.shape;
            if (!shape || !Array.isArray(shape)) {
                setError('A valid shape is required for tensor nodes');
                return;
            }
            
            try {
                const id = crypto.randomUUID();
                const node = {
                    id,
                    type: 'Tensor' as NodeType,
                    params: { shape }
                };

                addNode(node);

                // If this is an input tensor, make it a source
                if (params.isInput) {
                    makeTensorSource(id);
                }

                setOpType(null);
                setParams({});
                setRawInputs({});
                setError(null);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to create tensor node');
            }
            return;
        }
        
        // Determine node type based on opType prefix
        let nodeType: NodeType;
        let actualOpType = opType;
        
        if (opType.startsWith('Branch:')) {
            nodeType = 'Branch';
            actualOpType = opType.slice(7);
        } else if (opType.startsWith('Merge:')) {
            nodeType = 'Merge';
            actualOpType = opType.slice(6);
        } else {
            nodeType = 'Op';
        }
        
        // Validate parameters
        const validationError = validateModuleParams(params, actualOpType);
        if (validationError) {
            setError(validationError);
            return;
        }

        try {
            const id = crypto.randomUUID();
            
            addNode({
                id,
                type: nodeType,
                opType: actualOpType,
                params
            });
            
            setOpType(null);
            setParams({});
            setRawInputs({});
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to create node');
        }
    }, [opType, params, addNode, validateModuleParams, makeTensorSource]);

    const metadata = opType ? getMeta(opType === 'Tensor' ? 'Tensor' : 'Op', opType === 'Tensor' ? undefined : opType) : null;

    useEffect(() => {
        if (selectedId && selectedNodeData) {
            // Get data from the store's nodes array
            const type = selectedNodeData.data.type;
            const opType = selectedNodeData.data.opType;
            
            let displayType: string | null = null;
            if (type === 'Tensor') {
                displayType = 'Tensor';
            } else if (type === 'Branch') {
                displayType = `Branch:${opType}`;
            } else if (type === 'Merge') {
                displayType = `Merge:${opType}`;
            } else {
                displayType = opType || null;
            }
            
            setOpType(displayType);
            setParams(selectedNodeData.data.params || {});
            setRawInputs(paramsToRawInputs(selectedNodeData.data.params || {}));
            setError(null);
        } else if (!selectedId) {
            setOpType(null);
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
                value={opType}
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
                    disabled={!opType || !metadata || validateModuleParams(params, opType) !== null}
                    variant="filled"
                >
                    Add Layer
                </Button>
            )}
        </Box>
    );
}
