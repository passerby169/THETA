import { runApprovedThetaPlanCreate } from './tools/hypha-runner.js';

const result = await runApprovedThetaPlanCreate({
  plan: {
    datasetId: 'demo-dataset',
    modelId: 'lda',
    mode: 'unsupervised',
    numTopics: 8,
    textColumn: 'content',
  },
  rationale: 'Approved smoke test verifies Hypha approval before THETA state writes.',
});

if (result.status !== 'completed' || !result.output) {
  throw new Error(`approved theta.plan.create did not complete: ${JSON.stringify(result.error ?? result.status)}`);
}

console.log(
  JSON.stringify({
    status: 'ok',
    runner: 'GovernedToolRunner',
    toolId: result.toolId,
    planId: result.output.planId,
    valid: result.output.valid,
    approvalRequired: result.output.approvalRequired,
  })
);
