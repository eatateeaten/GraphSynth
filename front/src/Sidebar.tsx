import { useState, useCallback, useEffect } from 'react';
import { Select, Button, TextInput, Box, Text } from '@mantine/core';
import { CheckerNodes, CheckerNodeType, CheckerNodeConfig, CheckerNodeParams } from './checker';
import { useGraphStore } from './store';
import { ParamFieldMetadata } from './checker/node';
import { Shape } from './checker/shape';

/* Rearrange CheckerNodes into a format that can be used by the Select component */
const LAYER_TYPE_OPTIONS = Object.entries(CheckerNodes).reduce((groups, [type, node]) => {
    const meta = node.getMeta();
    const category = meta.category;
    const item = { value: type, label: meta.label };
    
    const existingGroup = groups.find(g => g.group === category);
    if (existingGroup) {
        existingGroup.items.push(item);
    } else {
        groups.push({ group: category, items: [item] });
    }
    
    return groups;
}, [] as Array<{ group: string; items: Array<{ value: string; label: string }> }>);

export function Sidebar() {
    const [type, setType] = useState<CheckerNodeType | null>(null);
    const [params, setParams] = useState<Partial<CheckerNodeParams>>({});
    const [rawInputs, setRawInputs] = useState<Record<string, string>>({});
    const [touchedFields, setTouchedFields] = useState<Set<string>>(new Set());
    const [error, setError] = useState<string | null>(null);
    const addNode = useGraphStore(state => state.addNode);
    const selectedId = useGraphStore(state => state.selectedId);
    const updateNodeParams = useGraphStore(state => state.updateNodeParams);
    const checkerNode = useGraphStore(state => selectedId ? state.checkerGraph.getNode(selectedId) : null);

    // Get default params for a node type
    const getDefaultParams = useCallback((nodeType: CheckerNodeType) => {
        const NodeClass = CheckerNodes[nodeType];
        const metadata = NodeClass.getMeta();
        const defaults: Partial<CheckerNodeParams> = {};
        
        Object.entries(metadata.paramFields).forEach(([name, field]) => {
            if (field.default !== undefined) {
                defaults[name] = field.default;
            }
        });

        return defaults;
    }, []);

    // Initialize raw inputs from params
    const paramsToRawInputs = useCallback((params: Partial<CheckerNodeParams>) => {
        const rawValues: Record<string, string> = {};
        Object.entries(params).forEach(([key, value]) => {
            rawValues[key] = value.toString();
        });
        return rawValues;
    }, []);

    // Reset raw inputs when type changes
    const handleTypeChange = (v: string | null) => {
        setType(v as CheckerNodeType);
        if (v) {
            const defaults = getDefaultParams(v as CheckerNodeType);
            setParams(defaults);
            setRawInputs(paramsToRawInputs(defaults));
        } else {
            setParams({});
            setRawInputs({});
        }
        setTouchedFields(new Set());
        setError(null);
    };

    const validateParams = useCallback((params: Partial<CheckerNodeParams>, type: CheckerNodeType): string | null => {
        const NodeClass = CheckerNodes[type];
        return NodeClass.validateParams(params);
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
        const error = showError && type ? validateParams(params, type) : undefined;

        switch (field.type) {
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
                        error={error}
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
                        error={error}
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
                        error={error}
                        onFocus={() => setTouchedFields(fields => new Set([...fields, name]))}
                    />
                );
        }
    };

    const handleAdd = useCallback(() => {
        if (!type || !params) return;
        
        const NodeClass = CheckerNodes[type];
        if (!NodeClass) return;

        const error = NodeClass.validateParams(params);
        if (error) {
            setError(error);
            return;
        }

        const config: CheckerNodeConfig = {
            type,
            params: params as CheckerNodeParams[typeof type]
        };

        try {
            const id = crypto.randomUUID();
            addNode(id, config);
            setType(null);
            setParams({});
            setRawInputs({});
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to create node');
        }
    }, [type, params, addNode]);

    const selectedNode = type && CheckerNodes[type];
    const metadata = selectedNode?.getMeta();

    useEffect(() => {
        if (selectedId && checkerNode) {
            setType(checkerNode.type as CheckerNodeType);
            setParams(checkerNode.params);
            setRawInputs(paramsToRawInputs(checkerNode.params));
            setError(null);
        } else if (!selectedId) {
            setType(null);
            setParams({});
            setRawInputs({});
            setError(null);
        }
    }, [selectedId, checkerNode, paramsToRawInputs]);

    return (
        <Box p="md" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <Select
                label="Layer Type"
                placeholder="Choose Layer..."
                data={LAYER_TYPE_OPTIONS}
                value={type}
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
                    disabled={!type || !metadata || validateParams(params, type) !== null}
                    variant="filled"
                >
                    Add Layer
                </Button>
            )}
        </Box>
    );
}
