import { InMemoryEventStore } from '@hypha/core';
import { GovernedToolRunner, type ToolCallContext, type ToolCallResult } from '@hypha/tools';
import { createThetaHyphaToolRegistry } from './hypha-registry.js';
import type { ThetaModelCatalogInput, ThetaModelCatalogOutput } from './model-catalog-tool.js';
import type { ThetaModelRecommendInput, ThetaModelRecommendOutput } from './model-recommend-tool.js';
import type { ThetaPlanCreateInput, ThetaPlanCreateOutput } from './plan-create-tool.js';
import type { ThetaPlanValidateInput, ThetaPlanValidateOutput } from './plan-validate-tool.js';
import { THETA_PERMISSION_SCOPES, THETA_TOOL_IDS } from './tool-ids.js';

export interface ThetaHyphaRunnerOptions {
  userId?: string;
  workspaceId?: string;
  permissionScopes?: string[];
  idempotencyKey?: string;
  invocationId?: string;
}

export interface ThetaHyphaRuntime {
  runner: GovernedToolRunner;
  trace: InMemoryEventStore;
}

export const createThetaHyphaRuntime = (): ThetaHyphaRuntime => {
  const registry = createThetaHyphaToolRegistry();
  const trace = new InMemoryEventStore();
  const runner = new GovernedToolRunner(registry, trace);
  return { runner, trace };
};

export const createThetaToolCallContext = (
  runId: string,
  stepId: string,
  options: ThetaHyphaRunnerOptions = {}
): ToolCallContext => ({
  runId,
  stepId,
  invocationId: options.invocationId,
  idempotencyKey: options.idempotencyKey,
  userId: options.userId ?? 'local_user',
  workspaceId: options.workspaceId ?? 'local_workspace',
  principal: {
    id: options.userId ?? 'local_user',
    type: 'user',
    userId: options.userId ?? 'local_user',
    workspaceId: options.workspaceId ?? 'local_workspace',
    permissionScopes: options.permissionScopes ?? [THETA_PERMISSION_SCOPES.modelRead],
  },
  metadata: {
    source: 'theta-cli-agent',
  },
});

export const runThetaModelCatalog = async (
  input: ThetaModelCatalogInput = {},
  options: ThetaHyphaRunnerOptions = {}
): Promise<ToolCallResult<ThetaModelCatalogOutput>> => {
  const { runner } = createThetaHyphaRuntime();
  return runner.run({
    toolId: THETA_TOOL_IDS.modelCatalog,
    input,
    context: createThetaToolCallContext('theta-model-catalog-smoke', 'model_catalog', options),
  }) as Promise<ToolCallResult<ThetaModelCatalogOutput>>;
};

export const runThetaModelRecommend = async (
  input: ThetaModelRecommendInput,
  options: ThetaHyphaRunnerOptions = {}
): Promise<ToolCallResult<ThetaModelRecommendOutput>> => {
  const { runner } = createThetaHyphaRuntime();
  return runner.run({
    toolId: THETA_TOOL_IDS.modelRecommend,
    input,
    context: createThetaToolCallContext('theta-model-recommend-smoke', 'model_recommend', {
      ...options,
      permissionScopes: options.permissionScopes ?? [
        THETA_PERMISSION_SCOPES.modelRead,
        THETA_PERMISSION_SCOPES.datasetRead,
      ],
    }),
  }) as Promise<ToolCallResult<ThetaModelRecommendOutput>>;
};

export const runThetaPlanValidate = async (
  input: ThetaPlanValidateInput,
  options: ThetaHyphaRunnerOptions = {}
): Promise<ToolCallResult<ThetaPlanValidateOutput>> => {
  const { runner } = createThetaHyphaRuntime();
  return runner.run({
    toolId: THETA_TOOL_IDS.planValidate,
    input,
    context: createThetaToolCallContext('theta-plan-validate-smoke', 'plan_validate', {
      ...options,
      permissionScopes: options.permissionScopes ?? [
        THETA_PERMISSION_SCOPES.planRead,
        THETA_PERMISSION_SCOPES.modelRead,
      ],
    }),
  }) as Promise<ToolCallResult<ThetaPlanValidateOutput>>;
};

export const requestThetaPlanCreate = async (
  input: ThetaPlanCreateInput,
  options: ThetaHyphaRunnerOptions = {}
): Promise<ToolCallResult<ThetaPlanCreateOutput>> => {
  const { runner } = createThetaHyphaRuntime();
  return runner.run({
    toolId: THETA_TOOL_IDS.planCreate,
    input,
    context: createThetaToolCallContext('theta-plan-create-approval-smoke', 'plan_create', {
      ...options,
      idempotencyKey: options.idempotencyKey ?? 'theta-plan-create-approval-smoke',
      permissionScopes: options.permissionScopes ?? [THETA_PERMISSION_SCOPES.planWrite],
    }),
  }) as Promise<ToolCallResult<ThetaPlanCreateOutput>>;
};

export const runApprovedThetaPlanCreate = async (
  input: ThetaPlanCreateInput,
  options: ThetaHyphaRunnerOptions = {}
): Promise<ToolCallResult<ThetaPlanCreateOutput>> => {
  const { runner } = createThetaHyphaRuntime();
  const invocationId = options.invocationId ?? 'theta-plan-create-approved-smoke';
  const context = createThetaToolCallContext('theta-plan-create-approved-smoke', 'plan_create', {
    ...options,
    invocationId,
    idempotencyKey: options.idempotencyKey ?? 'theta-plan-create-approved-smoke',
    permissionScopes: options.permissionScopes ?? [THETA_PERMISSION_SCOPES.planWrite],
  });
  const requested = await runner.run({
    toolId: THETA_TOOL_IDS.planCreate,
    input,
    context,
  });

  if (requested.status !== 'human_review_required') {
    return requested as ToolCallResult<ThetaPlanCreateOutput>;
  }

  return runner.approveAndResume(invocationId, options.userId ?? 'local_user') as Promise<
    ToolCallResult<ThetaPlanCreateOutput>
  >;
};
