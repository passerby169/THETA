# Hypha API Map

Hypha checkout: `CodeSoul-co/Hypha`, branch `dev-domain-merge`,
commit `dfb787ceed7d3c525fb39909142af7d1b5106c41`.

This map records the current APIs that THETA CLI Agent must use. Do not assume names from planning
documents when they differ from this checkout.

## Packages Required By The First Slice

Runtime imports currently require:

- `@hypha/core`
- `@hypha/tools`
- runtime dependencies: `zod`, `ajv`, `ajv-formats`

The CLI package imports Hypha packages through file dependencies:

- `@hypha/core`: `file:../Hypha/packages/core`
- `@hypha/tools`: `file:../Hypha/packages/tools`

## Core

Source: `Hypha/packages/core/src`.

Important exports:

- `JsonSchema`: shared JSON Schema type from `specs.ts`.
- `PolicyDecision`: `{ allowed, requiresHumanReview?, policyId?, ruleId?, reason?, metadata? }`.
- `PolicyEvaluationContext`: includes `runId`, `stepId`, `userId`, `capabilityId`,
  `sideEffectLevel`, `input`, and `metadata`.
- `PolicyEngine`: `evaluate(context): Promise<PolicyDecision>`.
- `allowAllPolicyEngine`, `denyExternalEffectsPolicyEngine`, `createPolicySpecEngine`.
- `TraceRecorder`: `record(event: FrameworkEvent): Promise<void>`.
- `EventStore`: `append(event)`, `list(filter?)`.
- `InMemoryEventStore`: implements both `EventStore` and `TraceRecorder`.
- `createFrameworkEvent(input)`: canonical Framework event constructor.

Initial THETA usage:

- Use `InMemoryEventStore` for the first local smoke slice.
- Replace with a durable local EventStore before TrainingPlan or TrainingRun authority moves out of
  Python.
- Use `PolicyEngine` directly. Do not create a parallel local policy abstraction.

## Tools

Source: `Hypha/packages/tools/src/index.ts`.

### ToolSpec

Required first-slice fields:

- `id`
- `version`
- `description`
- `inputSchema`
- `sideEffectLevel`

Common optional fields:

- `revision`
- `displayName`
- `outputSchema`
- `permissionScope`
- `timeoutPolicy`
- `retryPolicy`
- `auditPolicy`
- `humanApprovalPolicy`
- `idempotencyPolicy`
- `metadata`

### ToolRegistry

Constructor: `new ToolRegistry()`.

Methods:

- `register(spec, handler, options?)`
- `registerAdapter(spec, adapter, options?)`
- `unregister(toolId)`
- `getSpec(toolId)`
- `getAdapter(toolId)`
- `getTargetResolver(toolId)`
- `resolve({ id, version?, revision? })`
- `list()`

Registration options:

- `replace?: boolean`
- `targetResolver?: ToolTargetResolver`

### ToolAdapter

Interface:

- `id`
- `source`
- `capabilities()`
- `execute(request)`
- optional `cancel(request)`
- `health()`
- optional `close()`

Built-in local adapter:

- `LocalFunctionToolAdapter(id, handler)`

First THETA Bridge integration should implement `ToolAdapter` instead of calling the Python Bridge
from CLI or WorkflowExecutor.

### GovernedToolRunner

Constructor:

```ts
new GovernedToolRunner(
  registry,
  trace,
  policy = denyExternalEffectsPolicyEngine,
  options?
)
```

Options include:

- `approvalStore`
- `invocationStore`
- `authorizer`
- `middleware`
- `artifactPort`
- `snapshotStore`
- `receiptReconciler`
- `resultCache`
- `resultCacheFailureMode`
- `resultCacheTimeoutMs`
- `resultCacheMaxEntryBytes`
- `resultCacheArtifactVerifier`
- `observationPort`
- `telemetry`
- `now`

ToolRunner API:

- `run(request): Promise<ToolCallResult>`
- optional `cancelInvocation(invocationId, reason?)`

Default stores:

- `InMemoryToolApprovalStore`
- `InMemoryToolInvocationStore`

Default authorizer:

- `PermissionScopeToolAuthorizer`

Important implication:

- permission checks should be represented on `ToolSpec.permissionScope` or
  `ToolSpec.governance.requiredPermissionScopes`;
- side-effect policy flows through `PolicyEngine`;
- idempotency flows through `ToolInvocationStore`.

## FSM

Source: `Hypha/packages/fsm/src/index.ts`.

Important exports:

- `FSMProcessSpec`
- `FSMStateSpec`
- `FSMTransitionSpec`
- `FSMSnapshot`
- `FSMRuntimeOptions`
- `FSMGuardEvaluator`
- `validateFSMProcessSpec(spec)`
- `getAllowedTransitions(spec, stateId)`
- `createInitialSnapshot(spec, runId, now?)`
- `defaultReActFSMProcessSpec`

First THETA DomainPack should compile to `FSMProcessSpec`. Runtime code should persist and recover
`FSMSnapshot`; it should not infer state from CLI state.

## Domain

Source: `Hypha/packages/domain/src/index.ts`.

Important exports:

- `DomainPackSpec`
- `WorkflowSpec`
- `WorkflowStateSpec`
- `WorkflowTransitionSpec`
- `SessionProfileSpec`
- `DomainPackRegistry`
- `LocalDomainPackLoader`
- `DomainCompiler`
- `WorkflowCompiler`
- `initializeDomainSession(domainPack, options?)`
- `compileWorkflowToFSM(domainPack, options?)`
- `compileDomainPackToHarnessedSystem(input, options)`
- `validateDomainPackSpec(input)`

Workflow state bindings expose:

- `stateId`
- `allowedTools`
- `allowedSkills`
- `requiredSkills`
- optional tool profile refs and policy refs through compilation.

First THETA DomainPack should be minimal:

- `SessionInitialized`
- `DatasetInspection`
- `Completed`
- `Failed`

## Current Migration Guardrails

- CLI must not call `callThetaBridge` directly after the governed adapter exists.
- WorkflowExecutor must not spawn Python.
- Python Bridge must not decide FSM state, policy, approval, or recommendation authority.
- Formal contracts should import `JsonSchema` from `@hypha/core`.
- First tool migration should be `theta.model.catalog`; it is read-only and can validate registry,
  schema, permission, policy, event, and Bridge protocol without training side effects.
