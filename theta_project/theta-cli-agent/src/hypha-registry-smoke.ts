import { THETA_TOOL_IDS } from './tools/tool-ids.js';
import { createThetaHyphaToolRegistry } from './tools/hypha-registry.js';

const registry = createThetaHyphaToolRegistry();
const spec = registry.getSpec(THETA_TOOL_IDS.modelCatalog);

if (!spec) {
  throw new Error(`${THETA_TOOL_IDS.modelCatalog} was not registered.`);
}

console.log(
  JSON.stringify({
    status: 'ok',
    registry: 'ToolRegistry',
    toolId: spec.id,
    sideEffectLevel: spec.sideEffectLevel,
    permissionScope: spec.permissionScope,
  })
);
