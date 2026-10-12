'use strict';

// 仓库骨架与保密性校验（M0 起生效）。零依赖：只用 Node.js 内置模块。
// 不调用 child_process —— 受限沙箱里带管道的子进程会失败（spawn EPERM）。

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');
const readBytes = (rel) => fs.readFileSync(path.join(ROOT, rel));

// 这些目录不参与扫描：要么是生成物，要么是本地数据
const SKIP_DIRS = new Set([
  '.git',
  'node_modules',
  'data',
  '.tmp',
  '.venv',
  '__pycache__',
  '.pytest_cache',
]);

// 参与 UTF-8 校验的文本文件
const TEXT_EXTENSIONS = new Set([
  '.md',
  '.txt',
  '.json',
  '.js',
  '.py',
  '.ps1',
  '.sql',
  '.html',
  '.css',
  '.example',
  '.ini',
  '.cfg',
]);

// 代码块里的 # 是注释而非标题，统计标题前先剔除围栏代码块
function stripFences(text) {
  return text.replace(/^```[^\n]*\n[\s\S]*?^```[^\n]*$/gm, '');
}

function walk(relDir, out = []) {
  const abs = path.join(ROOT, relDir);
  for (const entry of fs.readdirSync(abs, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      walk(path.join(relDir, entry.name), out);
    } else {
      out.push(path.join(relDir, entry.name));
    }
  }
  return out;
}

test('需求与架构文档存在、非空、有唯一 H1 且包含关键章节', () => {
  const docs = [
    ['docs/需求规格说明书.md', ['需求规格', '功能需求', '反幻觉', '里程碑']],
    ['docs/架构说明.md', ['架构', '数据模型', '检索设计', 'LangGraph', '测试策略']],
  ];
  for (const [rel, keywords] of docs) {
    assert.ok(fs.existsSync(path.join(ROOT, rel)), `${rel} 必须存在`);
    const text = read(rel);
    assert.ok(text.trim().length > 200, `${rel} 内容过少`);
    const h1 = stripFences(text).match(/^#\s+\S.*$/gm) || [];
    assert.equal(h1.length, 1, `${rel} 应当只有一个一级标题，实际 ${h1.length} 个`);
    for (const kw of keywords) {
      assert.ok(text.includes(kw), `${rel} 必须包含「${kw}」相关章节`);
    }
  }
});

test('requirements.txt 覆盖项目关键依赖', () => {
  const text = read('requirements.txt');
  for (const dep of [
    'fastapi',
    'uvicorn',
    'langgraph',
    'langgraph-checkpoint-sqlite',
    'openai',
    'sqlite-vec',
    'pypdfium2',
    'pillow',
    'sentence-transformers',
    'pytest',
  ]) {
    assert.match(
      text,
      new RegExp(`^${dep}[><=]`, 'm'),
      `requirements.txt 缺少依赖 ${dep}`,
    );
  }
});

test('.env.example 提供配置模板且不含真实密钥', () => {
  const text = read('.env.example');
  for (const key of [
    'QWEN_API_KEY=',
    'QWEN_VISION_MODEL=',
    'OLLAMA_MODEL=',
    'EMBEDDING_MODEL_PATH=',
    'DB_PATH=',
  ]) {
    assert.ok(text.includes(key), `.env.example 缺少配置项 ${key}`);
  }
  assert.equal(
    /\bsk-[A-Za-z0-9_.-]{16,}/.test(text),
    false,
    '.env.example 不得包含真实密钥',
  );
});

test('.gitignore 覆盖密钥与本地运行数据', () => {
  const text = read('.gitignore');
  for (const pattern of ['.env', 'data/app.db', 'data/uploads/', 'data/docs/user/', '.tmp/', '.pytest_cache/']) {
    assert.ok(text.includes(pattern), `.gitignore 必须忽略 ${pattern}`);
  }
});

test('仓库内不存在真实 API 密钥', () => {
  const offenders = [];
  for (const rel of walk('.')) {
    if (path.basename(rel) === 'package-lock.json') continue;
    const ext = path.extname(rel).toLowerCase();
    const base = path.basename(rel);
    if (!TEXT_EXTENSIONS.has(ext) && base !== '.gitignore') continue;
    const text = read(rel);
    const hits = text.match(/\bsk-(?:ws-)?[A-Za-z0-9_.-]{16,}/g);
    if (hits) offenders.push(`${rel}: ${hits.length} 处`);
  }
  assert.deepEqual(offenders, [], `检测到疑似真实密钥：${offenders.join('; ')}`);
});

test('本地数据目录结构就绪', () => {
  assert.ok(fs.existsSync(path.join(ROOT, 'data/docs/samples/.gitkeep')), '缺少 data/docs/samples/');
  assert.ok(fs.existsSync(path.join(ROOT, 'data/docs/user/.gitkeep')), '缺少 data/docs/user/');
});

test('前端页面与静态资源完整（引用的文件都存在）', () => {
  const pages = ['app/web/index.html', 'app/web/admin.html', 'app/web/selftest.html'];
  for (const page of pages) {
    const html = read(page);
    assert.match(html, /<title>[^<]+<\/title>/, `${page} 缺少标题`);
    // 页面必须挂载到后端实际提供的静态路径
    const assets = [...html.matchAll(/(?:href|src)="(\/static\/[^"]+)"/g)].map((m) => m[1]);
    assert.ok(assets.length >= 2, `${page} 应当引用 CSS 与 JS 资源`);
    for (const asset of assets) {
      const relative = path.join('app/web', asset.replace('/static/', ''));
      assert.ok(fs.existsSync(path.join(ROOT, relative)), `${page} 引用了不存在的资源 ${asset}`);
    }
  }
  // 三个页面各自需要的挂载容器
  assert.ok(read('app/web/index.html').includes('id="messages"'), '对话页缺少消息容器');
  assert.ok(read('app/web/admin.html').includes('id="ticket-rows"'), '工单台缺少列表容器');
  assert.ok(read('app/web/selftest.html').includes('id="log"'), '自检页缺少输出容器');
});

test('客户界面不暴露内部工具入口（V3 权限边界）', () => {
  const customer = [read('app/web/index.html'), read('app/web/js/chat.js')].join('\n');

  for (const forbidden of ['接口文档', '/docs', '/admin', '工单台']) {
    assert.equal(
      customer.includes(forbidden),
      false,
      `客户界面不应出现「${forbidden}」（工单台与接口文档是内部工具）`,
    );
  }

  // 客服台与开发自检页仍然保留工单台入口
  assert.ok(read('app/web/admin.html').includes('href="/admin"'), '客服台应保留自身导航');
  assert.ok(read('app/web/selftest.html').includes('href="/admin"'), '自检页应保留工单台入口');
  // 接口文档入口在所有页面都移除（/docs 路由保留给开发，不在界面暴露）
  for (const page of ['app/web/index.html', 'app/web/admin.html', 'app/web/selftest.html']) {
    assert.equal(read(page).includes('接口文档'), false, `${page} 仍暴露接口文档入口`);
  }
});

test('转人工界面元素齐全（V3 F23-F25）', () => {
  const customer = read('app/web/index.html');
  assert.ok(customer.includes('id="btn-human"'), '客户页缺少「转人工客服」按钮');

  const admin = read('app/web/admin.html');
  for (const id of ['handoff-bar', 'handoff-list', 'handoff-waiting', 'conv-panel', 'conv-messages', 'conv-input', 'conv-send']) {
    assert.ok(admin.includes(`id="${id}"`), `工单台缺少转人工元素 #${id}`);
  }

  const chat = read('app/web/js/chat.js');
  assert.ok(chat.includes('requestHandoff'), '客户页缺少转人工逻辑');
  assert.ok(chat.includes('messagesAfter'), '客户页缺少客服消息轮询');
  assert.ok(chat.includes('stopPolling'), '缺少停止轮询逻辑');

  const adminJs = read('app/web/js/admin.js');
  assert.ok(adminJs.includes('loadHandoffs'), '工单台缺少转人工提醒加载');
  assert.ok(adminJs.includes('openConversationPanel'), '工单台缺少会话面板');
  assert.ok(adminJs.includes('sendAgentReply'), '工单台缺少客服回复逻辑');

  const api = read('app/web/js/api.js');
  for (const method of ['requestHandoff', 'handoffs', 'agentReply', 'messagesAfter']) {
    assert.ok(api.includes(method), `API 客户端缺少 ${method}`);
  }
});

test('前端脚本不拼接 innerHTML 渲染后端数据（防 XSS）', () => {
  for (const file of [
    'app/web/js/chat.js',
    'app/web/js/admin.js',
    'app/web/js/selftest.js',
  ]) {
    const text = read(file);
    assert.equal(
      /\.innerHTML\s*=/.test(text),
      false,
      `${file} 不应直接赋值 innerHTML（后端/模型文本必须走 textContent / DOM 节点）`,
    );
  }
});

test('仓库内文本文件均为有效 UTF-8 且不含 NUL 字节', () => {
  for (const rel of walk('.')) {
    const ext = path.extname(rel).toLowerCase();
    const base = path.basename(rel);
    if (!TEXT_EXTENSIONS.has(ext) && base !== '.gitignore') continue;
    const buf = readBytes(rel);
    assert.equal(buf.includes(0), false, `${rel} 含 NUL 字节`);
    assert.ok(
      Buffer.from(buf.toString('utf8'), 'utf8').equals(buf),
      `${rel} 不是有效的 UTF-8`,
    );
  }
});
