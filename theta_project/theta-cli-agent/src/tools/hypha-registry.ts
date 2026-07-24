import { ToolRegistry } from '@hypha/tools';
import { thetaModelCatalogHandler, thetaModelCatalogToolSpec } from './model-catalog-tool.js';

export const registerThetaModelCatalogTool = (registry: ToolRegistry): ToolRegistry => {
  registry.register(thetaModelCatalogToolSpec, thetaModelCatalogHandler, { replace: true });
  return registry;
};

export const createThetaHyphaToolRegistry = (): ToolRegistry => {
  const registry = new ToolRegistry();
  registerThetaModelCatalogTool(registry);
  return registry;
};
