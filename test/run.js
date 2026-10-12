'use strict';

// 依次加载同目录下所有 *.test.js。
//
// 这里刻意不用 `node --test`：测试运行器会为每个测试文件派生一个带管道的
// 子进程，在受限沙箱中会以 spawn EPERM 失败。用 require 在单进程内执行，
// node:test 同样会输出报告，并在断言失败时以非零退出码结束。

const fs = require('node:fs');
const path = require('node:path');

const dir = __dirname;
const files = fs
  .readdirSync(dir)
  .filter((name) => name.endsWith('.test.js'))
  .sort();

if (files.length === 0) {
  console.error(`未在 ${dir} 找到任何 *.test.js 测试文件`);
  process.exit(1);
}

for (const name of files) {
  require(path.join(dir, name));
}
