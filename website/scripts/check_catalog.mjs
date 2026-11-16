import assert from 'node:assert/strict';
import { readFile, stat, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const catalog = JSON.parse(await readFile(resolve(root, 'public/data/expert-catalog.json'), 'utf8'));
const archivalSources = process.argv.includes('--archival-sources');
const allowPending = process.argv.includes('--allow-pending');
assert.equal(catalog.tasks.length, 20);
assert.equal(new Set(catalog.tasks.map(t => t.id)).size, 20);
assert.equal(new Set(catalog.tasks.map(t => t.number)).size, 20);
assert(catalog.tasks.some(task => task.id === 'mimicx_stairs'));
assert(catalog.tasks.every(task => !catalog.excluded_from_display.includes(task.id)));
assert(catalog.tasks.every(task => !/^mimicx_(descent|bench|forest|football|tennis|dance|kungfu)$/.test(task.id)));
assert(catalog.tasks.every(task => !task.id.endsWith('_wall') || task.id.endsWith('_no_wall')));
assert.equal(catalog.public_sha256, '898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d');
assert.notEqual(catalog.public_sha256, catalog.excluded_base_sha256);
let recordings = 0;
const checked = new Set();
for (const task of catalog.tasks) {
  assert(catalog.groups.some(group => group.id === task.group));
  assert(task.label && task.detail);
  assert(!/HoloMotion|PPO|DAgger|蒸馏|MimicX|ATHLETE|HUSKY|ULTRA/i.test(`${task.label} ${task.detail}`));
  for (const mode of ['on', 'off']) {
    const record = task[mode];
    if (!record) { assert(allowPending, `${task.id} ${mode} recording missing`); continue; }
    recordings++;
    assert(Number.isFinite(record.duration) && record.duration > 0);
    for (const id of [record.id, record.fullId]) {
      assert(/^[a-z0-9_/-]+$/.test(id) && !id.includes('..'));
      for (const extension of ['mp4', 'webp', 'json']) await access(resolve(root, `public/media/${id}.${extension}`));
    }
    if (checked.has(record.id)) continue;
    checked.add(record.id);
    const file = resolve(root, `public/media/${record.id}.mp4`);
    const metadata = JSON.parse(await readFile(file.replace(/\.mp4$/, '.json'), 'utf8'));
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'stream=codec_name,pix_fmt:format=duration', '-of', 'json', file], { encoding: 'utf8' }));
    assert.equal(probe.streams[0].codec_name, 'h264');
    assert.equal(probe.streams[0].pix_fmt, 'yuv420p');
    assert(Math.abs(Number(probe.format.duration) - record.duration) < .08, record.id);
    const hash = createHash('sha256');
    for await (const block of createReadStream(file)) hash.update(block);
    assert.equal(hash.digest('hex'), metadata.video_sha256, record.id);
    if (/^(catalog31|catalog31-white-v2|gallery-bright-v3)\//.test(record.id)) {
      assert.equal(metadata.task, task.id, `${record.id}: selected skill must match the video`);
      assert.equal(metadata.expert_enabled, mode === 'on');
      assert.equal(metadata.public_sha256, catalog.public_sha256);
      assert.equal(metadata.teacher_queries, 0);
      assert.equal(metadata.training_updates, 0);
      assert.equal(metadata.speed, 1);
      assert.equal(metadata.pose_interpolation, false);
      const source = resolve(root, '..', metadata.source);
      if (archivalSources) assert((await stat(source)).isFile(), metadata.source);
      if (/^(catalog31-white-v2|gallery-bright-v3)\//.test(record.id)) {
        assert.equal(metadata.background_style, 'white-studio');
        assert.equal(metadata.all_native_geometry_retained, true);
      }
      if (record.id.startsWith('gallery-bright-v3/')) {
        assert.equal(metadata.lighting_profile, 'bright-robot-fill-v3');
        assert.equal(metadata.robot_only_lighting_pass, true);
        assert(metadata.headlight_ambient.every(value => value >= .61));
      }
    }
  }
}
const vite = await createServer({ root, configFile: false, server: { middlewareMode: true, watch: null }, appType: 'custom' });
try {
  const { default: Gallery } = await vite.ssrLoadModule('/src/ExpertGallery.jsx');
  const markup = renderToStaticMarkup(React.createElement(Gallery, {
    onOpenFilm() {}, Video: props => React.createElement('video', { 'data-recording': props.id }),
  }));
  assert.equal((markup.match(/role="switch"/g) || []).length, 1);
  assert.equal((markup.match(/role="tab"/g) || []).length, 20);
  assert(markup.includes('Search skills') && markup.includes('20 skills'));
  assert(markup.includes('aria-orientation="horizontal"') && markup.includes('Next skills'));
  const visibleText = markup.replace(/<[^>]+>/g, ' ');
  assert(!/\p{Script=Han}/u.test(markup), 'Gallery UI and accessibility labels must use English');
  assert(!/HoloMotion|PPO|DAgger|蒸馏|MimicX|ATHLETE|HUSKY|ULTRA/i.test(visibleText));
  for (const task of catalog.tasks) assert(markup.includes(task.label), task.label);
} finally { await vite.close(); }
console.log(`PASS: 20 selected skills; ${recordings}/40 recordings verified; excluded wall/tail tasks; one switch; horizontal selector; filters/search; source-free labels; server rendering. Browser interaction not exercised. Archival training logs are checked only with --archival-sources.`);
