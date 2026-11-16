"""Package this website, its built preview, and every referenced lossless master.

Run after npm run build and npm run portable. Does not include training checkpoints.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

WEBSITE = Path(__file__).resolve().parents[1]

README = """PERA 网站可编辑完整版 · __SNAPSHOT__

一、先完整解压，再打开
把整个压缩包复制到另一台电脑，右键“全部解压”。不要直接在压缩包里打开网页。
双击最外层“打开网站.html”，即可查看当前网站。请使用较新的 Chrome、Edge 或 Safari。
这是离线入口；网页、视频、封面和图表均从本文件夹读取，不需要这台电脑的原路径。
Paper 仍为 Coming soon；Code 保持原来的私有仓库链接，访问仓库本身需要网络和权限。
如果浏览器限制本地 HTML，安装 Node.js 后运行最外层“启动本地预览-Windows.bat”，
或在本文件夹运行 node tools/serve.mjs，再打开终端显示的本机地址。

二、继续编辑网站
1. 安装 Node.js 22.12 或更高的受支持版本（含 npm）。
2. 用 VS Code、Codex 或其他编辑器打开 website 文件夹。
3. 在 website 文件夹的终端执行：
   npm ci
   npm run dev -- --open
   首次 npm ci 需要联网下载锁定版本的依赖。之后本地编辑不需要训练环境、Python 或 GPU。
4. 编辑后浏览器自动更新；停止服务按 Ctrl+C。
5. 修改完成后执行：
   npm run build
   npm run portable
   前一条更新 dist 正式网页；后一条更新双击打开的 OPEN-WEBSITE.html。
   最外层“打开网站.html”会自动指向更新后的离线入口。

三、常改的地方
website/src/content.js       首页视频选择、Paper 和仓库链接
website/src/main.jsx         页面结构、概念图、播放器、交互
website/src/ExpertGallery.jsx 技能选择、分类搜索与唯一专家开关
website/src/styles.css       排版、颜色、尺寸、响应式布局
website/public/media/        全部 __VIDEO_COUNT__ 个 MP4、对应封面和素材来源 JSON（含子文件夹）
website/public/data/expert-catalog.json 当前 __SKILL_COUNT__ 项展示技能及开／关录像配置
website/public/concept-one-policy.svg 主对话最新概念图，内嵌图片，可独立打开和编辑
website/public/data/         网站使用的全部实验数据、CSV、图表 SVG
website/public/method.svg    方法示意图
website/public/favicon.svg   网站图标
website/index.html           页面标题、描述和源码入口
website/dist/                已构建的标准网站，可由本地 HTTP 服务或静态托管打开
website/OPEN-WEBSITE.html    已打包 JS/CSS 的离线入口，与 public 文件夹一起使用
video-masters/               当前素材引用的全部 __MASTER_COUNT__ 份 FFV1 无损 MKV 母版
MATERIALS.json               全部网页素材清单、大小，以及母版与网页视频的对应关系
website/docs/、licenses/     录制说明、验证记录和第三方许可
deployment/                 原 GitHub Pages 工作流参考；请按原仓库目录结构使用

更换视频：将 H.264 / yuv420p 的 MP4 和同名 WebP 封面放进 public/media，
同步更新 content.js、比较数据和素材来源 JSON；保持动作原速并说明剪辑范围。
无损 MKV 母版用于重新剪辑，浏览网页不需要播放这些 MKV。
示例（另需安装 FFmpeg；start 和 length 以秒为单位）：
ffmpeg -ss start -i input.master.mkv -t length -an -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart output.mp4

四、当前页面版本
首页为 31 秒四段原生场景录像：滑板 5 秒 → 不平路/高台阶 5 秒 → 上楼梯 6 秒 → 网球近景 15 秒。
技能模块展示 __SKILL_COUNT__ 项技能、__PAIRED_COUNT__ 组专家开／关录像；技能选择为单行横向滚动。
有墙任务及末尾七项已移出展示；补充了连续上楼梯录像，地形走舞的机器人已调亮。
第一部分使用主对话最新概念图和一句说明；项目名称统一为 PERA。
只有网球动作持拍；Paper 维持 Coming soon。

五、完整性与范围
本包包括当前网站源码、所有网页素材、构建版本、离线入口和所有被当前素材元数据引用的无损母版。
无需原 ModuMotion-g1 工程即可查看、修改网页、替换素材和重新构建。
为保证跨系统使用，未复制仅适用于本机的 node_modules；依赖版本由 package-lock.json 固定。
这不是机器人训练工程备份，不含模型权重、训练数据全集或旧版未使用的场景实验。
scripts 中保留原始录制/仿真重渲染脚本供参考；它们会引用原研究工程和环境。
来源 JSON 中的 reports/、.render-work/ 等路径是历史来源记录，不是网页运行依赖。
重新剪辑现有素材直接使用本包 video-masters/；重新运行机器人仿真仍需原研究工程。

传输后可在最外层执行 node tools/verify.mjs 校验全部文件。
文件主动修改后，原校验值不同是正常现象。检查结果只用于确认传输完整性。
"""

SERVER = r"""import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../website/dist');
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8',
 '.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8',
 '.svg':'image/svg+xml','.webp':'image/webp','.png':'image/png','.jpg':'image/jpeg',
 '.jpeg':'image/jpeg','.mp4':'video/mp4','.webm':'video/webm','.pdf':'application/pdf',
 '.csv':'text/csv; charset=utf-8'};
const server = createServer(async (req, res) => {
 try {
  const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = resolve(root, '.' + (path.endsWith('/') ? path + 'index.html' : path));
  if (!file.startsWith(root + sep)) { res.writeHead(403).end(); return; }
  const info = await stat(file);
  if (!info.isFile()) { res.writeHead(404).end(); return; }
  const headers = {'Content-Type':mime[extname(file)] || 'application/octet-stream',
   'Accept-Ranges':'bytes', 'Cache-Control':'no-cache'};
  let start = 0, end = info.size - 1, status = 200;
  if (req.headers.range) {
   const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
   if (!match || (!match[1] && !match[2])) { res.writeHead(416, {'Content-Range':`bytes */${info.size}`}).end(); return; }
   if (!match[1]) start = Math.max(0, info.size - Number(match[2]));
   else { start = Number(match[1]); if (match[2]) end = Math.min(end, Number(match[2])); }
   if (start > end || start >= info.size) { res.writeHead(416, {'Content-Range':`bytes */${info.size}`}).end(); return; }
   status = 206;
   headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
  }
  headers['Content-Length'] = end - start + 1;
  res.writeHead(status, headers);
  if (req.method === 'HEAD') { res.end(); return; }
  const stream = createReadStream(file, {start, end});
  stream.on('error', () => res.destroy());
  res.on('close', () => stream.destroy());
  stream.pipe(res);
 } catch { if (!res.headersSent) res.writeHead(404); res.end('Not found'); }
});
const i = process.argv.indexOf('--port');
server.listen(i >= 0 ? Number(process.argv[i+1]) : 0, '127.0.0.1', () => {
 console.log(`PERA: http://127.0.0.1:${server.address().port}/`);
 console.log('Open this address in your browser. Press Ctrl+C to stop.');
});
"""

VERIFY = r"""import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(await readFile(resolve(root, 'CHECKSUMS.json'), 'utf8'));
let failed = 0;
for (const entry of manifest.files) {
 try {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(resolve(root, entry.path))) hash.update(chunk);
  if (hash.digest('hex') !== entry.sha256) throw new Error('checksum mismatch');
 } catch (error) { console.error('FAIL', entry.path, error.message); failed++; }
}
console.log(`${failed ? 'FAIL' : 'PASS'}: ${manifest.files.length} files, ${failed} errors`);
process.exitCode = failed ? 1 : 0;
"""


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--zip', dest='archive', type=Path, required=True)
    parser.add_argument('--name', default='PERA-Website-Editable-2026-11-06')
    parser.add_argument('--snapshot', default='2026-11-06')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise SystemExit('Package name must be a single directory name')
    root = args.work / args.name
    if root.exists() or args.archive.exists():
        raise SystemExit('Refusing to overwrite an existing package')
    if not (WEBSITE / 'OPEN-WEBSITE.html').exists() or not (WEBSITE / 'dist/index.html').exists():
        raise SystemExit('Run npm run build and npm run portable first')
    root.mkdir(parents=True)
    dest = root / 'website'
    dest.mkdir()
    for name in ['src', 'public', 'dist', 'docs', 'licenses', 'scripts', 'tools']:
        shutil.copytree(WEBSITE / name, dest / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['package.json', 'package-lock.json', 'vite.config.js', 'index.html',
                 'README.md', '.gitignore', '.gitattributes', 'OPEN-WEBSITE.html']:
        if (WEBSITE / name).exists():
            shutil.copy2(WEBSITE / name, dest / name)
    # Preserve every current public asset, including older gallery and complete recordings.
    assets = [{'path': p.relative_to(WEBSITE).as_posix(), 'bytes': p.stat().st_size}
              for p in sorted((WEBSITE / 'public').rglob('*')) if p.is_file()]
    masters = {}
    for path in sorted((WEBSITE / 'public').rglob('*.json')):
        data = json.loads(path.read_text())
        for value in strings(data):
            if value.endswith('.master.mkv'):
                source = Path(value)
                if not source.is_absolute():
                    source = (WEBSITE.parent / value if value.startswith('website/')
                              else WEBSITE / value)
                if not source.is_file():
                    raise FileNotFoundError(f'Referenced master is missing: {value}')
                relative = source.relative_to(WEBSITE / '.render-work')
                target = root / 'video-masters' / relative
                key = target.relative_to(root).as_posix()
                if key not in masters:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    masters[key] = {'path': key, 'original_path': value,
                                    'bytes': source.stat().st_size, 'used_by': []}
                masters[key]['used_by'].append(path.relative_to(WEBSITE).as_posix())
    video_count=len(list((dest / 'public/media').rglob('*.mp4')))
    catalog=json.loads((dest/'public/data/expert-catalog.json').read_text())
    (root / 'MATERIALS.json').write_text(json.dumps({
        'web_assets': assets, 'lossless_masters': list(masters.values()),
        'web_video_count': video_count,
        'displayed_skill_count':len(catalog['tasks']),
        'displayed_pair_count':catalog['counts']['pairedVideos'],
        'snapshot':args.snapshot,
        'scope': 'Complete editable website, runtime assets, builds, and all currently referenced lossless masters. Not a training-project backup.'
    }, indent=2, ensure_ascii=False) + '\n')
    instructions=README
    for key,value in {'SNAPSHOT':args.snapshot,'VIDEO_COUNT':video_count,
                      'SKILL_COUNT':len(catalog['tasks']),'PAIRED_COUNT':catalog['counts']['pairedVideos'],
                      'MASTER_COUNT':len(masters)}.items():
        instructions=instructions.replace(f'__{key}__',str(value))
    (root / '先读我-使用说明.txt').write_text(instructions, encoding='utf-8-sig')
    (root / '打开网站.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=website/OPEN-WEBSITE.html"><title>打开 PERA 网站</title><p><a href="website/OPEN-WEBSITE.html">点击打开 PERA 网站</a>（请先完整解压）</p></html>''')
    (root / 'tools').mkdir()
    (root / 'tools/serve.mjs').write_text(SERVER)
    (root / 'tools/verify.mjs').write_text(VERIFY)
    (root / '启动本地预览-Windows.bat').write_bytes(('''@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed. Open the offline HTML instead.
  start "" "website\\OPEN-WEBSITE.html"
  pause
  exit /b 0
)
node tools\\serve.mjs
pause
''').replace('\n', '\r\n').encode())
    launcher = root / '启动本地预览-macOS-Linux.command'
    launcher.write_text('''#!/bin/sh
cd "$(dirname "$0")" || exit 1
node tools/serve.mjs
''')
    launcher.chmod(0o755)
    workflow = WEBSITE.parent / '.github/workflows/pear-pages.yml'
    if workflow.exists():
        (root / 'deployment').mkdir()
        shutil.copy2(workflow, root / 'deployment/pear-pages.yml')
    manifest = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                 'sha256': sha256(p)} for p in sorted(root.rglob('*')) if p.is_file()]
    (root / 'CHECKSUMS.json').write_text(json.dumps({'algorithm': 'SHA-256', 'files': manifest}, indent=2) + '\n')
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', allowZip64=True, compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                compression = (zipfile.ZIP_STORED if path.suffix.lower() in
                               {'.mp4', '.mkv', '.webp', '.png', '.jpg'} else zipfile.ZIP_DEFLATED)
                z.write(path, (Path(root.name) / path.relative_to(root)).as_posix(), compress_type=compression)
    print(json.dumps({'archive': str(args.archive), 'bytes': args.archive.stat().st_size,
                      'files': len(manifest) + 1, 'web_assets': len(assets),
                      'videos': video_count,
                      'displayed_skills':len(catalog['tasks']),
                      'masters': len(masters)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
