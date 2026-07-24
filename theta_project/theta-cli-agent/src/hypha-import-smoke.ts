import { ToolRegistry } from '@hypha/tools';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export function createHyphaToolRegistrySmoke(): ToolRegistry {
  return new ToolRegistry();
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const registry = createHyphaToolRegistrySmoke();
  console.log(JSON.stringify({ status: 'ok', registry: registry.constructor.name }));
}
