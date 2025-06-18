import { ModuleDB } from './db';

// Import all module definitions using import *
import * as basic from './basic';
import * as linear from './linear';
import * as pooling from './pooling';
import * as reshape from './reshape';
import * as normalization from './normalization';
import * as dropout from './dropout';
import * as activation from './activation';
import * as convolutional from './convolutional';
import * as merge from './merge';
import * as branch from './branch';
import { ModuleDef } from './types';

// Collect all module files
const moduleFiles = [
    basic,
    linear,
    pooling,
    reshape,
    normalization,
    dropout,
    activation,
    convolutional,
    merge,
    branch
];

// Auto-register all modules from all files
moduleFiles.forEach(moduleFile => {
    Object.values(moduleFile).forEach(module => {
        ModuleDB.register(module as ModuleDef);
    });
});

// Export the populated database
export { ModuleDB };
export type { ModuleDef, ParamDef } from './types';
