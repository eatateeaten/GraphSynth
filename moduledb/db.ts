import { ModuleDef } from './types';

class ModuleDatabase {
    private modules = new Map<string, ModuleDef>();
    
    register(name: string, spec: ModuleDef): void {
        if (this.modules.has(name)) {
            throw new Error(`Module ${name} already registered`);
        }
        this.modules.set(name, spec);
    }
    
    get(name: string): ModuleDef {
        const spec = this.modules.get(name);
        if (!spec) throw new Error(`Module ${name} not found`);
        return spec;
    }
    
    has(name: string): boolean {
        return this.modules.has(name);
    }
    
    getAll(): Map<string, ModuleDef> {
        return new Map(this.modules);
    }
    
    getAllNames(): string[] {
        return Array.from(this.modules.keys());
    }
    
    getByCategory(category: string): Record<string, ModuleDef> {
        const result: Record<string, ModuleDef> = {};
        for (const [name, spec] of this.modules) {
            if (spec.category === category) {
                result[name] = spec;
            }
        }
        return result;
    }
    
    getAllCategories(): string[] {
        const categories = new Set<string>();
        for (const spec of this.modules.values()) {
            categories.add(spec.category);
        }
        return Array.from(categories);
    }
    
    validateParams(moduleName: string, params: Record<string, any>): string | null {
        try {
            const spec = this.get(moduleName);
            const missing: string[] = [];
            
            for (const [paramName, paramSpec] of Object.entries(spec.params)) {
                if (paramSpec.required && !(paramName in params)) {
                    missing.push(paramName);
                }
            }
            
            if (missing.length > 0) {
                return `Missing required parameters: ${missing.join(', ')}`;
            }
            
            return null;
        } catch (e: any) {
            return e.message;
        }
    }
    
    generateCode(moduleName: string, params: Record<string, any>): string {
        const spec = this.get(moduleName);
        return spec.toPytorchModule(params);
    }
    
    inferShape(moduleName: string, inShape: number[], params: Record<string, any>): number[] {
        const spec = this.get(moduleName);
        return spec.inferOutputShape(inShape, params);
    }
}

export const ModuleDB = new ModuleDatabase();
