import { nn_module_metadata } from '../isomorphic/torch_nn_module_op';
import { allModules } from '../app/registry';

describe('Registry Tests', () => {
  // Helper function to get all operation names from the registry
  function getAllRegistryOps(): string[] {
    return Object.keys(allModules)
      .filter(key => key.startsWith('Op:'))
      .map(key => key.slice(3)); // Remove 'Op:' prefix
  }

  // Helper function to get all operation names from torch_nn_module_op
  function getAllTorchOps(): string[] {
    return Object.keys(nn_module_metadata);
  }

  test('Registry operations match torch_nn_module_op operations', () => {
    const registryOps = getAllRegistryOps();
    const torchOps = getAllTorchOps();

    // Check that all registry ops exist in torch ops
    const missingInTorch = registryOps.filter(op => !torchOps.includes(op));
    expect(missingInTorch).toEqual([]);

    // Check that all torch ops exist in registry
    const missingInRegistry = torchOps.filter(op => !registryOps.includes(op));
    expect(missingInRegistry).toEqual([]);
  });

  test('Registry operation parameters match torch_nn_module_op parameters', () => {
    const registryOps = getAllRegistryOps();

    for (const op of registryOps) {
      // Get registry metadata
      const registryMeta = allModules[`Op:${op}`];

      // Get torch metadata
      const torchMeta = nn_module_metadata[op];

      // Check that required parameters match
      const registryParams = Object.keys(registryMeta.paramFields || {});
      const torchRequiredParams = torchMeta.required_params;
      const torchOptionalParams = torchMeta.optional_params;

      // All required parameters from torch should be in registry
      const missingRequired = torchRequiredParams.filter(param => !registryParams.includes(param));
      expect(missingRequired).toEqual([]);

      // All parameters in registry should be either required or optional in torch
      const extraParams = registryParams.filter(param => 
        !torchRequiredParams.includes(param) && !torchOptionalParams.includes(param)
      );
      expect(extraParams).toEqual([]);
    }
  });
});
