import { ModuleDB } from './db';

// Import all module definitions
import { Tensor } from './basic';
import { Linear, Identity } from './linear';
import { Conv2D } from './convolutional';
import { ReLU } from './activation';
// TODO: Import from other files as they're populated

// Explicit registration - this ensures modules are registered in a predictable order
// Basic nodes
ModuleDB.register('Tensor', Tensor);

// Linear operations
ModuleDB.register('Linear', Linear);
ModuleDB.register('Identity', Identity);

// Convolutional operations
ModuleDB.register('Conv2D', Conv2D);

// Activation operations
ModuleDB.register('ReLU', ReLU);

// TODO: Register modules from other categories

// Export the populated database
export { ModuleDB };
export type { ModuleDef as UnifiedModuleSpec, ParamDef as ParamFieldSpec } from './types';
