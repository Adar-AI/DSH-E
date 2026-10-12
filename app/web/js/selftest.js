/**
 * 前端自检页逻辑（app/web/selftest.html）。
 *
 * 用途：不打开开发者工具也能确认「浏览器 ↔ 后端 ↔ 模型」这条链路是否通畅——
 * 依次检查健康状态、知识库、SSE 流式对话（事件序列与首包耗时）、工单接口。
 * 访问 /static/selftest.html 点击「开始自检」，或直接 /static/selftest.html?auto=1 自动跑。
 */
(() => {
  const el = UI.el;
  const dom = {};

  document.addEventListener('DOMContentLoaded', () => {
    dom.log = document.getElementById('log');
    dom.run = document.getElementById('btn-run');
    dom.question = document.getElementById('question');
    dom.summary = document.getElementById('summary');
    dom.skipChat = document.getElementById('skip-chat');

    dom.run.addEventListener('click', run);

    const params = new URLSearchParams(window.location.search);
    // ?auto=1 自动跑；&chat=0 跳过需要等模型的流式对话检查（快速自检连通性）
    if (params.get('chat') === '0') dom.skipChat.checked = true;
    if (params.get('auto')) {
      dom.question.value = params.get('q') || dom.question.value;
      run();
    }
  });

  function log(text, kind = '') {
    dom.log.append(el('div', { class: `log-line ${kind}`.trim(), text }));
    dom.log.scrollTop = dom.log.scrollHeight;
  }

  function heading(text) {
    dom.log.append(el('div', { class: 'log-head', text }));
  }

  /** 步骤②：真实跑一轮流式对话，统计事件序列与首包耗时。 */
  async function checkStreaming(problems) {
    const question = dom.question.value.trim() || '打印机报错E05怎么处理';
    log(`提问：${question}`);
    const counts = {};
    let answer = '';
    let citations = [];
    let firstTokenAt = null;
    const started = performance.now();

    await API.streamChat({
      text: question,
      onEvent: ({ name, data }) => {
        counts[name] = (counts[name] || 0) + 1;
        if (name === 'meta') log(`会话：${data.conversation_id}`);
        else if (name === 'vision') log(`图片识别：ok=${data.ok}`);
        else if (name === 'retrieval') {
          citations = data.chunks || [];
          log(`检索：命中=${data.hit} 最高分=${data.best_score} 模式=${data.mode} 支撑片段=${citations.length}`);
        } else if (name === 'token') {
          if (firstTokenAt === null) firstTokenAt = performance.now() - started;
          answer += data.text;
        } else if (name === 'draft') {
          log(`工单草稿字段：${Object.keys(data.draft || {}).join('、') || '（空）'}`);
        } else if (name === 'error') {
          problems.push(`服务端返回错误：${data.message}`);
          log(`错误：${data.message}`, 'error');
        }
      },
    });

    const total = performance.now() - started;
    log(`事件统计：${Object.entries(counts).map(([key, value]) => `${key}×${value}`).join('，')}`);
    log(`首包 ${firstTokenAt === null ? '—' : `${firstTokenAt.toFixed(0)}ms`}，整轮 ${total.toFixed(0)}ms`);
    if (!counts.token) problems.push('没有收到任何 token 事件（流式链路异常）');
    if (!counts.done) problems.push('没有收到 done 事件（本轮未正常结束）');
    if (!citations.length) log('本次未命中知识库，走的是兜底话术', 'warn');
    log(`回答预览：${answer.slice(0, 160) || '（空）'}`);
  }

  async function run() {
    UI.clear(dom.log);
    dom.run.disabled = true;
    dom.summary.textContent = '自检中…';
    const started = performance.now();
    const problems = [];
    try {
      heading('① 服务与知识库');
      const health = await API.health();
      const channel = (health.chat.channels || [])[0] || {};
      log(`服务状态：${health.status}`);
      log(`对话通道：${channel.name || '—'} · ${channel.model || '—'}（配置${channel.available ? '可用' : '不可用'}）`);
      const vision = health.vision || {};
      log(
        `多模态通道：${vision.model}（${vision.kind || '—'}${vision.available ? '，可用' : `，${vision.reason}`}）` +
          (vision.local ? ' ← 本地识别' : ' ← 云端识别'),
      );
      if ((vision.channels || []).length > 1) {
        log(`视觉回退通道：${vision.channels.slice(1).map((item) => `${item.model}（${item.kind}）`).join('、')}`);
      }
      log(`检索模式：${health.retrieval.mode}，阈值 ${health.retrieval.min_score}`);
      log(`知识库：${health.knowledge.documents} 份文档 / ${health.knowledge.chunks} 个分块 / ${health.knowledge.vectors} 条向量`);
      if (!health.knowledge.documents) {
        problems.push('知识库为空，请先运行 scripts\\ingest_samples.py 入库资料');
        log('⚠️ 知识库为空，对话会全部走兜底话术', 'warn');
      }
      if (!channel.available) problems.push(`对话通道不可用：${channel.reason || ''}`);

      heading('② SSE 流式对话（真实调用一次）');
      if (dom.skipChat.checked) {
        log('已跳过流式对话自检（勾选了「跳过模型调用」）');
      } else {
        await checkStreaming(problems);
      }

      heading('③ 工单接口');
      const tickets = await API.tickets({ limit: 5 });
      log(`工单总数：${tickets.total}（${Object.entries(tickets.stats || {}).map(([k, v]) => `${k} ${v}`).join('，')}）`);
      if (tickets.items && tickets.items.length) {
        const first = await API.ticket(tickets.items[0].id);
        log(`最新工单：${first.ticket_no} · ${first.status} · 事件 ${(first.events || []).length} 条`);
      } else {
        log('还没有工单，可在对话页结束一次会话生成', 'warn');
      }

      const total = performance.now() - started;
      if (problems.length) {
        dom.summary.textContent = `自检完成，发现 ${problems.length} 个问题（${total.toFixed(0)}ms）`;
        dom.summary.className = 'badge badge-danger';
        problems.forEach((item) => log(`❌ ${item}`, 'error'));
      } else {
        dom.summary.textContent = `全部检查通过（${total.toFixed(0)}ms）`;
        dom.summary.className = 'badge badge-success';
        log('✅ 浏览器 ↔ 后端 ↔ 模型链路正常', 'ok');
      }
    } catch (error) {
      dom.summary.textContent = '自检失败';
      dom.summary.className = 'badge badge-danger';
      log(`❌ ${error.message}`, 'error');
    } finally {
      dom.run.disabled = false;
    }
  }
})();
