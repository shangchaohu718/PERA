// Produce a file://-friendly preview. All editable assets stay in public/.
import { build } from 'vite';
import { writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const result = await build({
  root,
  base: './',
  define: { 'import.meta.env.BASE_URL': JSON.stringify('./public/') },
  publicDir: false,
  build: { write: false, sourcemap: false },
});
const outputs = Array.isArray(result) ? result.flatMap(r => r.output) : result.output;
const page = outputs.find(o => o.fileName === 'index.html');
if (!page) throw new Error('Vite did not produce index.html');
let html = String(page.source);
for (const output of outputs) {
  if (output.type === 'chunk' && output.isEntry) {
    if (output.imports.length || output.dynamicImports.length) {
      throw new Error('Offline preview requires a single bundled entry');
    }
    html = html.replace(/<script\b[^>]*src="[^"]+"[^>]*><\/script>/,
      () => `<script type="module">${output.code.replace(/<\/script/gi, '<\\/script')}</script>`);
  }
  if (output.fileName.endsWith('.css')) {
    html = html.replace(/<link\b[^>]*rel="stylesheet"[^>]*>/,
      () => `<style>${String(output.source).replace(/<\/style/gi, '<\\/style')}</style>`);
  }
}
html = html.replace('href="./favicon.svg"', 'href="./public/favicon.svg"');
if (/<script\b[^>]*src=|<link\b[^>]*rel="stylesheet"/.test(html)) {
  throw new Error('External code dependencies remain in the offline preview');
}
await writeFile(resolve(root, 'OPEN-WEBSITE.html'), html);
console.log('Created OPEN-WEBSITE.html. Keep it next to public/; double-click to view offline.');
