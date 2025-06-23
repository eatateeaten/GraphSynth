import { ModuleDef } from './types';

class ModuleDatabase {
    private modules = new Map<string, ModuleDef>();
    
    register(spec: ModuleDef): void {
        const name = spec.label;
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
    
   
     //TODO invoke the generateCode from GraphNode class specific behaviors 

    //TODO Query for the helper function using the moduleName, and pass that to the function in the specific class inheriting GraphNode 
    // invoke the generateCode from GraphNode class specific behaviors 
    
    
    //TODO invoke the generateCode from GraphNode class specific behaviors 

}

export const ModuleDB = new ModuleDatabase();
