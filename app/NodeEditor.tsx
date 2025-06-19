import { useState, useCallback, useEffect } from 'react';
import { Select, Button, TextInput, Box, Text, Checkbox } from '@mantine/core';
import { v4 as uuidv4 } from "uuid";

import { useStore } from './store';
import { Shape } from '../OpCompiler/types';
import { ParamDef } from '../moduledb/types';
import { ModuleDB } from '../moduledb';

/* Build layer type options from available modules */
const LAYER_TYPE_OPTIONS = (() => {
    const modules = ModuleDB.getAll();
    const categories = new Map<string, Array<{ value: string; label: string }>>();
    
    for (const [name, module] of modules) {
        const category = module.category;
        if (!categories.has(category)) {
            categories.set(category, []);
        }
        categories.get(category)!.push({ value: name, label: module.label });
    }
    
    return Array.from(categories.entries()).map(([category, items]) => ({
        group: category,
        items
    }));
})();

export function NodeEditor() {
    const [moduleName, setModuleName] = useState<string | null>(null);
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
    const getDefaultParams = useCallback((moduleName: string) => {
        const module = ModuleDB.get(moduleName);
        const defaults: Record<string, any> = {};
        
        Object.entries(module.params).forEach(([name, field]) => {
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
        setModuleName(v);
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

    const validateModuleParams = useCallback((params: Record<string, any>, moduleName: string): string | null => {
        return ModuleDB.validateParams(moduleName, params);
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

    const renderParamField = (name: string, field: ParamDef) => {
        const value = params[name];
        const rawValue = rawInputs[name] ?? '';
        const showError = touchedFields.has(name);
        const validationError = showError && moduleName ? validateModuleParams(params, moduleName) : undefined;

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
        if (!moduleName) return;
        
        try {
            const module = ModuleDB.get(moduleName);
            
            // Validate parameters
            const validationError = validateModuleParams(params, moduleName);
            if (validationError) {
                setError(validationError);
                return;
            }

            if (["Op", "PointwiseOp"].includes(module.moduleType)) {
                params.opType = moduleName;
            }

            // Use the module's moduleType to determine what kind of node to create
            addNode({
                id: uuidv4(),
                type: module.moduleType,
                moduleName,  // Store the module name for editing later
                params
            });
            
            setModuleName(null);
            setParams({});
            setRawInputs({});
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to create node');
        }
    }, [moduleName, params, addNode, validateModuleParams]);

    // Get metadata for the selected module key
    const metadata = moduleName ? ModuleDB.get(moduleName) : null;

    useEffect(() => {
        if (selectedId && selectedNodeData) {
            // Use the stored module name from node data
            setModuleName(selectedNodeData.data.moduleName || null);
            setParams(selectedNodeData.data.params || {});
            setRawInputs(paramsToRawInputs(selectedNodeData.data.params || {}));
            setError(null);
        } else if (!selectedId) {
            setModuleName(null);
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
                value={moduleName}
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
                    {Object.entries(metadata.params).map(([name, field]) => 
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
                    disabled={!moduleName || !metadata || validateModuleParams(params, moduleName) !== null}
                    variant="filled"
                >
                    Add Layer
                </Button>
            )}
        </Box>
    );
}
