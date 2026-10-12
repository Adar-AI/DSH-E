'use strict';

// 仓库文档完整性校验。零依赖：只用 Node.js 内置的 node:test / node:assert。
// 不调用 child_process —— 在受限沙箱里带管道的子进程会失败（spawn EPERM）。

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const REPO_URL = 'https://github.com/Adar-AI/DSH-E';

const read = (rel) => fs.readFileSync(path.join(ROOT, rel), 'utf8');

test('AGENTS.md 存在、非空，且包含两条强制约定', () => {
  const text = read('AGENTS.md');
  assert.ok(text.trim().length > 0, 'AGENTS.md 不应为空');
  assert.match(
    text,
    /每次改动完成[，,]\s*都必须创建一个对应的\s*Git commit/,
    'AGENTS.md 必须要求每次改动都创建对应的 commit',
  );
  assert.match(
    text,
    /确保所有测试和验证全部通过/,
    'AGENTS.md 必须要求在交付前确保测试与验证全部通过',
  );
});

test('README.md 有标题、指向真实远程地址、无残留占位符', () => {
  const text = read('README.md');
  assert.match(text, /^#\s+\S/m, 'README.md 必须有 Markdown 标题');
  assert.ok(text.includes(REPO_URL), `README.md 必须包含远程地址 ${REPO_URL}`);

  const leftover = text.match(/<[^<>\n]*(用户名|username|your[-_ ]?name)[^<>\n]*>/i);
  assert.equal(leftover, null, `README.md 不应残留占位符，发现 ${leftover?.[0]}`);
});

test('.gitignore 存在且非空', () => {
  assert.ok(read('.gitignore').trim().length > 0, '.gitignore 不应为空');
});

test('仓库内的文本文件均为有效 UTF-8 且不含 NUL 字节', () => {
  for (const rel of ['README.md', 'AGENTS.md', '.gitignore']) {
    const buf = fs.readFileSync(path.join(ROOT, rel));
    assert.equal(buf.includes(0), false, `${rel} 含 NUL 字节`);
    assert.ok(
      Buffer.from(buf.toString('utf8'), 'utf8').equals(buf),
      `${rel} 不是有效的 UTF-8`,
    );
  }
});
