import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const website = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = resolve(website, '..');
const repository = 'shangchaohu718/PERA';
const url = 'https://shangchaohu718.github.io/PERA/';

function command(bin, args, options = {}) {
  const result = spawnSync(bin, args, {
    cwd: repo, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024, ...options,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${bin} failed: ${result.stderr || result.stdout || result.status}`);
  }
  return result.stdout?.trim() || '';
}

const origin = command('git', ['remote', 'get-url', 'origin']);
if (!/github\.com[:/]shangchaohu718\/PERA(?:\.git)?\/?$/i.test(origin)) {
  throw new Error('Run deployment from the dedicated PERA checkout.');
}
command('gh', ['api', `repos/${repository}`, '--jq', '.permissions.push']);
// Confirm hosting is available before publishing any static branch.
command('gh', ['api', `repos/${repository}/pages`]);
command(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build'], {
  cwd: website, env: { ...process.env, PAGES_BASE_PATH: '/PERA/' }, stdio: 'inherit',
});
const dist = join(website, 'dist');
writeFileSync(join(dist, '.nojekyll'), '');

// A separate index stages only dist; the editable source checkout stays intact.
const temporary = mkdtempSync(join(tmpdir(), 'pera-pages-'));
try {
  const gitDirectory = command('git', ['rev-parse', '--absolute-git-dir']);
  const args = [`--git-dir=${gitDirectory}`, `--work-tree=${dist}`];
  const options = { cwd: dist, env: { ...process.env, GIT_INDEX_FILE: join(temporary, 'index') } };
  command('git', [...args, 'read-tree', '--empty'], options);
  command('git', [...args, 'add', '--all', '--force', '--', '.'], options);
  const tree = command('git', [...args, 'write-tree'], options);
  const remote = command('git', ['ls-remote', 'origin', 'refs/heads/gh-pages']);
  const parents = [];
  if (remote) {
    command('git', ['fetch', 'origin', 'gh-pages']);
    parents.push('-p', command('git', ['rev-parse', 'FETCH_HEAD']));
  }
  const commit = command('git', ['commit-tree', tree, ...parents], {
    input: 'Deploy PERA project website\n',
  });
  command('git', ['push', 'origin', `${commit}:refs/heads/gh-pages`], { stdio: 'inherit' });
  command('git', ['update-ref', 'refs/remotes/origin/gh-pages', commit]);
  command('gh', ['api', `repos/${repository}/pages`, '--method', 'PUT', '--input', '-'], {
    input: JSON.stringify({ build_type: 'legacy', source: { branch: 'gh-pages', path: '/' } }),
  });
  command('gh', ['api', `repos/${repository}/pages/builds`, '--method', 'POST', '--input', '-'], {
    input: '{}',
  });
  console.log(`Deployment requested: ${url}`);
  console.log(`Published commit: ${commit}`);
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
