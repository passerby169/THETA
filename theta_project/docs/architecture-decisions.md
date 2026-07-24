# THETA CLI Agent Architecture Decisions

## System Authority Boundaries

| Data | Authority |
| --- | --- |
| Session, Run, FSM snapshot | Hypha Runtime |
| Tool invocation, Policy, Approval, Idempotency | Hypha Runtime |
| Agent audit events | Hypha EventStore |
| ResearchBrief, DatasetProfile, TrainingPlan | TypeScript Runtime structured store |
| Python algorithm output | Python Bridge |
| Training PID, process status, logs | Python Runner |
| Model result files | Artifact Store / THETA result directory |
| Recommendation rules | TypeScript Recommendation Engine |
| Recommendation evidence | Local RAG |
| Natural-language wording | Deterministic templates first; Minimax only behind provider boundary |

No fact should have two independent owners. Python may return observed process or algorithm state,
but TypeScript Runtime owns Agent approvals, FSM state, tool policy, and event projection.

## Python Bridge Boundary

Python Bridge may:

- read and inspect local data through approved paths;
- call existing THETA Python capabilities;
- construct and validate executable training commands;
- manage local training processes through the Runner;
- read model results and local algorithm outputs.

Python Bridge must not:

- decide FSM state;
- decide user permissions or policy outcome;
- own Agent Session, Run, or conversation state;
- approve plans or training;
- call Minimax or any LLM provider.

## TypeScript Runtime Boundary

TypeScript Runtime owns:

- DomainPack, WorkflowSpec, FSM, and state guards;
- JSON Schema contracts;
- Policy, approval, idempotency, and event emission;
- artifact references and structured projections;
- research intake, conflict detection, recommendation, and plan generation;
- CLI orchestration.

Presentation or CLI code must not directly spawn Python or bypass the governed tool runner.

## TrainingPlan Authority

TrainingPlan is created by TypeScript Runtime. Python supplies catalog, dataset observations,
dry-run facts, and command feasibility checks.

Each TrainingPlan must bind:

- ResearchBrief;
- DatasetProfile and dataset hash;
- ColumnConfirmation;
- RecommendationResult;
- DomainPack and recommendation versions;
- canonical planHash.

Python `training_plans` and `plan_approvals` tables are transitional only and must not become
the authority for Agent approval.

## Event Authority

Agent events are owned by Hypha EventStore. Python Runner may surface process states such as
`training.started`, `training.progress`, `training.completed`, and `training.failed`; TypeScript
Runtime converts those observations into Hypha events.

The Python `agent_events` table is a transitional local log and must not be used as the final
Agent event source of truth.
