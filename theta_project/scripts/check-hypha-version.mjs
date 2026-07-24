import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const hyphaRoot = path.join(projectRoot, 'Hypha');
const lockPath = path.join(projectRoot, 'hypha.lock.json');

const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
const failures = [];
const warnings = [];

function check(condition, message, fix) {
  if (!condition) failures.push({ message, fix });
}

function warn(condition, message) {
  if (!condition) warnings.push(message);
}

function git(args) {
  return execFileSync('git', args, {
    cwd: hyphaRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function gitSucceeds(args) {
  try {
    execFileSync('git', args, {
      cwd: hyphaRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    return true;
  } catch {
    return false;
  }
}

function readPackageJson(packageName) {
  const relative = packageName.replace('@hypha/', '');
  const packagePath = path.join(hyphaRoot, 'packages', relative, 'package.json');
  if (!fs.existsSync(packagePath)) return null;
  return JSON.parse(fs.readFileSync(packagePath, 'utf8'));
}

check(fs.existsSync(hyphaRoot), 'Hypha checkout is missing.', 'Clone CodeSoul-co/Hypha into theta_project/Hypha.');
check(
  fs.existsSync(path.join(hyphaRoot, '.git')),
  'Hypha checkout is not a Git repository.',
  'Re-clone Hypha into theta_project/Hypha.'
);

if (failures.length === 0) {
  const remote = git(['remote', 'get-url', 'origin']);
  const branch = git(['branch', '--show-current']);
  const head = git(['rev-parse', 'HEAD']);
  const hasStagedChanges = !gitSucceeds(['diff', '--cached', '--quiet']);
  const hasContentChanges = !gitSucceeds(['diff', '--quiet', '--ignore-cr-at-eol']);
  const untracked = git(['ls-files', '--others', '--exclude-standard']);

  check(
    remote === lock.repository,
    `Hypha remote mismatch: expected ${lock.repository}, got ${remote}.`,
    `git -C theta_project/Hypha remote set-url origin ${lock.repository}`
  );
  check(
    branch === lock.branch,
    `Hypha branch mismatch: expected ${lock.branch}, got ${branch}.`,
    `git -C theta_project/Hypha switch ${lock.branch}`
  );
  check(
    head === lock.commit,
    `Hypha HEAD mismatch: expected ${lock.commit}, got ${head}.`,
    `git -C theta_project/Hypha fetch origin ${lock.branch} && git -C theta_project/Hypha switch ${lock.branch} && git -C theta_project/Hypha reset --hard ${lock.commit}`
  );
  check(
    !hasStagedChanges && !hasContentChanges && untracked.length === 0,
    'Hypha working tree is not clean.',
    'Commit, stash, or discard Hypha local changes before running THETA Agent.'
  );

  for (const packageName of lock.requiredPackages) {
    const packageJson = readPackageJson(packageName);
    const packageDir = packageName.replace('@hypha/', '');
    const distIndex = path.join(hyphaRoot, 'packages', packageDir, 'dist', 'index.js');
    check(packageJson !== null, `Required Hypha package is missing: ${packageName}.`, 'Refresh the Hypha checkout.');
    if (packageJson) {
      const expectedVersion = lock.packageVersions?.[packageName];
      warn(
        expectedVersion === undefined || packageJson.version === expectedVersion,
        `${packageName} version differs from lock: expected ${expectedVersion}, got ${packageJson.version}.`
      );
    }
    check(
      fs.existsSync(distIndex),
      `Hypha package build output is missing: ${packageName}/dist/index.js.`,
      'Run: npm --prefix theta_project/Hypha run build:packages'
    );
  }
}

for (const message of warnings) {
  console.warn(`WARN ${message}`);
}

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`FAIL ${failure.message}`);
    console.error(`FIX  ${failure.fix}`);
  }
  process.exit(1);
}

console.log(`PASS Hypha lock verified at ${lock.branch}@${lock.commit}`);
