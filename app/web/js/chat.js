/**
 * 客户对话页逻辑（app/web/index.html）。
 *
 * 事件流处理对应后端 app/api/chat.py：
 *   meta（会话确定）→ stage（阶段提示）→ vision（图片识别）→ retrieval（依据）
 *   → token（流式正文）→ draft（工单草稿）→ done（收尾）/ error
 *
 * 会话 ID 存在 localStorage，刷新页面后会自动恢复历史消息（需求 F1）。
 */
(() => {
  const el = UI.el;

  const STORAGE_KEY = 'dsh.conversationId';
  const EXAMPLE_QUESTIONS = [
    '我的打印机报错 E05，刚换了新硒鼓还是不识别，怎么处理？',
    '打印机频繁卡纸，纸是新买的，一天卡好几次',
    '这台机器保修期是多久？人为损坏算不算？',
    '无线网络连不上，路由器是 5G 频段的',
  ];

  const state = {
    conversationId: null,
    streaming: false,
    imageFile: null,
    controller: null,
  };

  const dom = {};

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    dom.messages = document.getElementById('messages');
    dom.input = document.getElementById('input-text');
    dom.send = document.getElementById('btn-send');
    dom.attach = document.getElementById('btn-attach');
    dom.file = document.getElementById('input-file');
    dom.thumb = document.getElementById('thumb');
    dom.convTitle = document.getElementById('conv-title');
    dom.convId = document.getElementById('conv-id');
    dom.close = document.getElementById('btn-close-conv');
    dom.newChat = document.getElementById('btn-new');
    dom.historyBtn = document.getElementById('btn-history');
    dom.historyDrawer = document.getElementById('history-drawer');
    dom.historyList = document.getElementById('history-list');
    dom.historyMask = document.getElementById('history-mask');
    dom.historyClose = document.getElementById('history-close');
    dom.infoId = document.getElementById('info-id');
    dom.infoCount = document.getElementById('info-count');
    dom.infoChannel = document.getElementById('info-channel');
    dom.infoMode = document.getElementById('info-mode');
    dom.infoStatus = document.getElementById('info-status');
    dom.draftBody = document.getElementById('draft-body');
    dom.draftCount = document.getElementById('draft-count');
    dom.draftBar = document.getElementById('draft-bar');
    dom.badgeChat = document.getElementById('status-chat');
    dom.badgeRetrieval = document.getElementById('status-retrieval');
    dom.badgeKb = document.getElementById('status-kb');

    dom.send.addEventListener('click', send);
    dom.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        send();
      }
    });
    dom.attach.addEventListener('click', () => dom.file.click());
    dom.file.addEventListener('change', onPickImage);
    dom.newChat.addEventListener('click', startNewConversation);
    dom.close.addEventListener('click', closeConversation);
    dom.historyBtn.addEventListener('click', openHistory);
    dom.historyClose.addEventListener('click', closeHistory);
    dom.historyMask.addEventListener('click', closeHistory);

    renderEmptyState();
    refreshHealth();

    // 支持深链 /?conv=<会话ID>（工单台的「来源会话」会链到这里）
    const requested = new URLSearchParams(window.location.search).get('conv');
    if (requested) {
      switchConversation(requested);
    } else {
      restoreConversation();
    }
  }

  /* ------------------------------------------------------------ 顶栏状态 */

  async function refreshHealth() {
    try {
      const health = await API.health();
      const channel = (health.chat.channels || [])[0];
      if (channel) {
        dom.badgeChat.textContent = `对话通道 ${channel.name} · ${channel.model}`;
        dom.badgeChat.className = `badge badge-dot ${channel.available ? 'badge-success' : 'badge-danger'}`;
      }
      dom.badgeRetrieval.textContent = `检索 ${health.retrieval.mode}`;
      dom.badgeKb.textContent = `知识库 ${health.knowledge.documents} 份 / ${health.knowledge.chunks} 分块`;
      dom.infoChannel.textContent = channel ? `${channel.name} · ${channel.model}` : '—';
      dom.infoMode.textContent =
        health.retrieval.mode === 'hybrid' ? 'hybrid（关键词 + 向量）' : 'keyword_only（仅关键词）';
      dom.badgeKb.className = health.knowledge.documents > 0 ? 'badge badge-muted' : 'badge badge-danger';
      if (health.knowledge.documents === 0) {
        UI.toast('知识库为空，请先运行 scripts\\ingest_samples.py 入库资料', 'error');
      }
    } catch (error) {
      dom.badgeChat.textContent = '服务不可用';
      dom.badgeChat.className = 'badge badge-dot badge-danger';
      UI.toast(`无法连接后端：${error.message}`, 'error');
    }
  }

  /* ------------------------------------------------------------ 消息渲染 */

  function scrollToBottom() {
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  function renderEmptyState() {
    UI.clear(dom.messages);
    const examples = el('div', { class: 'example-list' });
    EXAMPLE_QUESTIONS.forEach((question) => {
      examples.append(
        el('button', {
          type: 'button',
          text: question,
          onclick: () => {
            dom.input.value = question;
            dom.input.focus();
          },
        }),
      );
    });
    dom.messages.append(
      el('div', { class: 'empty-chat' }, [
        el('h3', { text: '描述你的问题，或上传故障截图' }),
        el('p', {
          class: 'muted',
          text: '我会先查企业内部资料（产品手册 / 售后规则 / 排查表），只依据资料作答，资料没有的会明确说明并建议转人工。',
        }),
        examples,
      ]),
    );
  }

  function avatar(role) {
    return el('div', { class: 'avatar', text: role === 'user' ? '客' : 'AI' });
  }

  /** 追加一条消息，返回其内部节点句柄，便于流式更新。 */
  function appendMessage(role, options = {}) {
    const note = el('div', { class: 'note-block', hidden: '' });
    const bubble = el('div', { class: 'msg-bubble' });
    const citations = el('div');
    const meta = el('div', { class: 'msg-meta' });
    const body = el('div', { class: 'msg-body' }, [note, bubble, citations, meta]);

    if (role === 'user' && options.imageUrl) {
      body.prepend(
        el('div', { class: 'msg-image' }, [el('img', { src: options.imageUrl, alt: '客户上传的图片' })]),
      );
    }
    if (options.text) bubble.textContent = options.text;

    const node = el('div', { class: `msg ${role}` }, [avatar(role), body]);
    // 空状态占位需要清掉
    const empty = dom.messages.querySelector('.empty-chat');
    if (empty) UI.clear(dom.messages);
    dom.messages.append(node);
    scrollToBottom();

    return { node, body, note, bubble, citations, meta };
  }

  function setNote(handle, text, kind = '') {
    if (!text) {
      handle.note.hidden = true;
      handle.note.textContent = '';
      return;
    }
    handle.note.hidden = false;
    handle.note.className = `note-block ${kind}`.trim();
    handle.note.textContent = text;
  }

  function setStage(handle, text) {
    handle.bubble.classList.add('streaming');
    UI.clear(handle.bubble);
    handle.bubble.append(el('span', { class: 'stage-note', text }));
  }

  function renderHistoryMessages(items) {
    UI.clear(dom.messages);
    if (!items.length) {
      renderEmptyState();
      return;
    }
    items.forEach((item) => {
      const handle = appendMessage(item.role === 'user' ? 'user' : 'assistant', {
        text: item.role === 'user' ? item.text : '',
        imageUrl: item.image_path ? `/uploads/${String(item.image_path).split(/[\\/]/).pop()}` : null,
      });
      if (item.role !== 'user' && item.text) {
        UI.renderRich(handle.bubble, item.text);
      }
      if (Array.isArray(item.citations) && item.citations.length) {
        const block = UI.citationsBlock(
          item.citations.map((citation) => ({ citation, text: '' })),
          { open: false },
        );
        if (block) handle.citations.append(block);
      }
      const trace = item.trace || {};
      if (item.role !== 'user') {
        const bits = [UI.formatShortTime(item.created_at)];
        if (trace.provider) bits.push(`通道 ${trace.provider}`);
        if (trace.hit === true) bits.push('命中知识库');
        if (trace.fallback_used) bits.push('未命中，已兜底');
        handle.meta.textContent = bits.join(' · ');
      } else {
        handle.meta.textContent = UI.formatShortTime(item.created_at);
      }
    });
    scrollToBottom();
  }

  /* ------------------------------------------------------------ 会话管理 */

  function updateHeader(conversation) {
    if (!conversation) {
      dom.convTitle.textContent = '新会话';
      dom.convId.textContent = '未开始';
      dom.close.disabled = true;
      dom.infoId.textContent = '—';
      dom.infoStatus.textContent = '未开始';
      return;
    }
    dom.convTitle.textContent = conversation.title || '新会话';
    dom.convId.textContent = String(conversation.id || '').slice(0, 18);
    dom.close.disabled = conversation.status === 'closed';
    dom.infoId.textContent = String(conversation.id || '').slice(0, 18);
    dom.infoCount.textContent = String(conversation.message_count ?? '—');
    dom.infoStatus.textContent = conversation.status === 'closed' ? '已结束' : '进行中';
    dom.infoStatus.className = conversation.status === 'closed' ? 'badge badge-muted' : 'badge badge-success';
  }

  async function restoreConversation() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      const conversation = await API.conversation(saved);
      state.conversationId = saved;
      const data = await API.messages(saved);
      renderHistoryMessages(data.items || []);
      updateHeader(conversation);
      renderDraft(conversation.ticket_draft || {});
    } catch (_) {
      localStorage.removeItem(STORAGE_KEY);
      state.conversationId = null;
    }
  }

  function startNewConversation() {
    state.conversationId = null;
    localStorage.removeItem(STORAGE_KEY);
    renderEmptyState();
    updateHeader(null);
    renderDraft({});
    dom.input.value = '';
    clearImage();
    UI.toast('已开始新会话');
  }

  async function openHistory() {
    dom.historyDrawer.hidden = false;
    dom.historyMask.hidden = false;
    UI.clear(dom.historyList).append(el('div', { class: 'loading', text: '加载中…' }));
    try {
      const data = await API.conversations(50);
      renderHistoryList(data.items || []);
    } catch (error) {
      UI.clear(dom.historyList).append(el('div', { class: 'loading', text: error.message }));
    }
  }

  function closeHistory() {
    dom.historyDrawer.hidden = true;
    dom.historyMask.hidden = true;
  }

  function renderHistoryList(items) {
    UI.clear(dom.historyList);
    if (!items.length) {
      dom.historyList.append(el('div', { class: 'loading', text: '还没有历史会话' }));
      return;
    }
    items.forEach((item) => {
      const active = item.id === state.conversationId;
      const row = el('div', { class: `history-item ${active ? 'active' : ''}`.trim() }, [
        el('div', { class: 'title', text: item.title || '未命名会话' }),
        el('div', { class: 'meta' }, [
          `${item.message_count ?? 0} 条消息 · ${UI.formatShortTime(item.updated_at)} · ${item.status === 'closed' ? '已结束' : '进行中'}`,
        ]),
      ]);
      const actions = el('div', { class: 'meta', style: 'margin-top:6px' });
      actions.append(
        el('button', {
          class: 'btn btn-sm',
          type: 'button',
          text: '打开',
          onclick: (event) => {
            event.stopPropagation();
            switchConversation(item.id);
          },
        }),
        el('button', {
          class: 'btn btn-sm btn-ghost',
          type: 'button',
          text: '删除',
          style: 'margin-left:6px',
          onclick: async (event) => {
            event.stopPropagation();
            if (!window.confirm(`删除会话「${item.title || '未命名'}」？工单不会被删除。`)) return;
            try {
              await API.deleteConversation(item.id);
              UI.toast('会话已删除');
              if (item.id === state.conversationId) startNewConversation();
              openHistory();
            } catch (error) {
              UI.toast(error.message, 'error');
            }
          },
        }),
      );
      row.append(actions);
      row.addEventListener('click', () => switchConversation(item.id));
      dom.historyList.append(row);
    });
  }

  async function switchConversation(id) {
    try {
      const conversation = await API.conversation(id);
      const data = await API.messages(id);
      state.conversationId = id;
      localStorage.setItem(STORAGE_KEY, id);
      renderHistoryMessages(data.items || []);
      updateHeader(conversation);
      renderDraft(conversation.ticket_draft || {});
      closeHistory();
    } catch (error) {
      UI.toast(error.message, 'error');
    }
  }

  /* ------------------------------------------------------------ 图片选择 */

  function onPickImage() {
    const file = dom.file.files && dom.file.files[0];
    if (!file) return;
    const allowed = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp'];
    if (!allowed.includes(file.type)) {
      UI.toast('只支持 PNG / JPG / WEBP / GIF / BMP 图片', 'error');
      dom.file.value = '';
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      UI.toast('图片超过 8MB，请压缩后再上传', 'error');
      dom.file.value = '';
      return;
    }
    state.imageFile = file;
    renderThumb();
  }

  function renderThumb() {
    UI.clear(dom.thumb);
    if (!state.imageFile) {
      dom.thumb.hidden = true;
      return;
    }
    dom.thumb.hidden = false;
    dom.thumb.append(
      el('span', {
        text: `🖼 ${state.imageFile.name}（${(state.imageFile.size / 1024).toFixed(1)} KB）`,
      }),
      el('button', {
        class: 'btn btn-ghost btn-sm',
        type: 'button',
        text: '移除',
        onclick: clearImage,
      }),
    );
  }

  function clearImage() {
    state.imageFile = null;
    dom.file.value = '';
    renderThumb();
  }

  function setBusy(busy) {
    dom.send.disabled = busy;
    dom.send.textContent = busy ? '生成中…' : '发送';
    dom.input.disabled = busy;
  }

  /* ------------------------------------------------------------ 发送与流式 */

  async function send() {
    if (state.streaming) return;
    const text = dom.input.value.trim();
    if (!text && !state.imageFile) {
      UI.toast('请输入问题或上传一张图片');
      return;
    }

    const imageFile = state.imageFile;
    const imageUrl = imageFile ? URL.createObjectURL(imageFile) : null;
    appendMessage('user', { text, imageUrl });
    if (imageUrl) setTimeout(() => URL.revokeObjectURL(imageUrl), 60000);

    dom.input.value = '';
    clearImage();
    state.streaming = true;
    setBusy(true);

    const handle = appendMessage('assistant');
    setStage(handle, '正在处理…');

    let answer = '';
    let chunks = [];
    let visionNote = '';
    let visionFailed = false;

    try {
      await API.streamChat({
        text,
        conversationId: state.conversationId,
        image: imageFile,
        onEvent: ({ name, data }) => {
          if (name === 'meta') {
            state.conversationId = data.conversation_id;
            localStorage.setItem(STORAGE_KEY, data.conversation_id);
            updateHeader({ id: data.conversation_id, title: data.title, status: data.status });
          } else if (name === 'stage') {
            if (!answer) setStage(handle, data.message || '处理中…');
          } else if (name === 'vision') {
            if (data.ok) {
              const where = data.local ? '本地模型' : '云端网关';
              visionNote = `图片识别结果（${where}：${data.model || data.provider || '未知'}）：\n${data.summary}`;
              setNote(handle, visionNote);
            } else {
              visionFailed = true;
              setNote(handle, `图片识别失败：${data.error || '未知原因'}，将按文字描述处理。`, 'warn');
            }
          } else if (name === 'retrieval') {
            chunks = data.chunks || [];
            if (!answer) {
              setStage(
                handle,
                data.hit
                  ? `已找到 ${chunks.length} 条相关资料，正在生成回答…`
                  : '内部资料未命中，正在整理建议…',
              );
            }
          } else if (name === 'token') {
            if (!answer) UI.clear(handle.bubble);
            answer += data.text;
            UI.renderRich(handle.bubble, answer);
            scrollToBottom();
          } else if (name === 'draft') {
            renderDraft(data.draft || {});
          } else if (name === 'done') {
            finalize(handle, data, chunks, visionNote, visionFailed);
          } else if (name === 'error') {
            handle.bubble.classList.remove('streaming');
            UI.clear(handle.bubble);
            handle.bubble.textContent = data.message || '服务异常';
            handle.bubble.classList.add('note-block', 'error');
          }
        },
      });
    } catch (error) {
      handle.bubble.classList.remove('streaming');
      UI.clear(handle.bubble);
      handle.bubble.textContent = `请求失败：${error.message}`;
      UI.toast(error.message, 'error');
    } finally {
      state.streaming = false;
      setBusy(false);
      dom.input.focus();
      refreshConversationInfo();
    }
  }

  function finalize(handle, done, chunks, visionNote, visionFailed) {
    handle.bubble.classList.remove('streaming');
    const answer = done.answer || handle.bubble.textContent || '';
    UI.renderRich(handle.bubble, answer);
    if (visionNote && !visionFailed) setNote(handle, visionNote);

    const block = UI.citationsBlock(chunks);
    if (block) {
      UI.clear(handle.citations).append(block);
    } else if (!done.trace?.hit) {
      handle.citations.append(
        el('div', { class: 'note-block warn', text: '本次回答未命中企业内部资料，建议转人工客服跟进。' }),
      );
    }

    const bits = [UI.formatShortTime(new Date().toISOString())];
    if (done.trace?.provider) bits.push(`通道 ${done.trace.provider}`);
    if (done.trace?.vision_ok) {
      bits.push(done.trace.vision_local ? '图片本地识别' : '图片云端识别');
    }
    bits.push(done.trace?.hit ? '命中知识库' : '未命中，已兜底');
    if (done.trace?.retrieval_mode === 'keyword_only') bits.push('仅关键词检索');
    if (done.answer_error) bits.push(`模型告警：${UI.truncate(done.answer_error, 40)}`);
    handle.meta.textContent = bits.join(' · ');

    if (done.ticket_draft) renderDraft(done.ticket_draft);
    scrollToBottom();
  }

  async function refreshConversationInfo() {
    if (!state.conversationId) return;
    try {
      const conversation = await API.conversation(state.conversationId);
      updateHeader(conversation);
      const data = await API.messages(state.conversationId);
      dom.infoCount.textContent = String((data.items || []).length);
      if (conversation.ticket_draft) renderDraft(conversation.ticket_draft);
    } catch (_) {
      /* 静默：信息栏刷新失败不影响对话 */
    }
  }

  /* ------------------------------------------------------------ 工单草稿 */

  const CORE_FIELDS = ['customer_request', 'product_model', 'fault_phenomenon', 'error_info', 'tried_actions', 'urgency'];

  function renderDraft(draft) {
    const data = draft || {};
    const filled = CORE_FIELDS.filter((key) => String(data[key] || '').trim()).length;
    dom.draftCount.textContent = `${filled} / ${CORE_FIELDS.length}`;
    dom.draftBar.style.width = `${Math.round((filled / CORE_FIELDS.length) * 100)}%`;

    UI.clear(dom.draftBody);
    const order = [...CORE_FIELDS, 'suggestion'];
    order.forEach((key) => {
      const raw = String(data[key] || '').trim();
      const value = UI.stripMarkdown(raw);
      const long = value.length > 120;
      dom.draftBody.append(
        el('div', { class: 'draft-field' }, [
          el('dt', { text: UI.FIELD_LABELS[key] || key }),
          el('dd', {
            class: value ? (long ? 'clamp' : '') : 'empty',
            title: long ? value : null,
            text: value || '（对话中未提到）',
          }),
        ]),
      );
    });
    if (Array.isArray(data.citations) && data.citations.length) {
      dom.draftBody.append(
        el('div', { class: 'draft-field' }, [
          el('dt', { text: '来源' }),
          el('dd', { text: `${data.citations.length} 条资料` }),
        ]),
      );
    }
  }

  /* ------------------------------------------------------------ 结束会话 */

  async function closeConversation() {
    if (!state.conversationId) return;
    if (!window.confirm('结束本次会话并生成工单？生成后可在工单台查看与流转。')) return;
    dom.close.disabled = true;
    try {
      const ticket = await API.closeConversation(state.conversationId);
      appendTicketCard(ticket);
      const conversation = await API.conversation(state.conversationId);
      updateHeader(conversation);
      UI.toast(`工单 ${ticket.ticket_no} 已生成`, 'success');
    } catch (error) {
      dom.close.disabled = false;
      UI.toast(error.message, 'error');
    }
  }

  function appendTicketCard(ticket) {
    const source = ticket.extraction?.source === 'llm' ? 'AI 结构化抽取' : '规则抽取（模型不可用）';
    const filled = Array.isArray(ticket.extraction?.filled)
      ? `${ticket.extraction.filled[0]}/${ticket.extraction.filled[1]}`
      : '—';
    const card = el('div', { class: 'panel', style: 'align-self:center;max-width:520px;width:100%' }, [
      el('div', { class: 'panel-title' }, [
        el('span', { text: `工单已生成 · ${ticket.ticket_no}` }),
        UI.statusEl(ticket.status),
      ]),
      el('div', { class: 'panel-body' }, [
        UI.fieldRow('客户诉求', ticket.customer_request),
        UI.fieldRow('产品型号', ticket.product_model, { empty: '（未识别）' }),
        UI.fieldRow('报错信息', ticket.error_info, { empty: '（未识别）' }),
        UI.fieldRow('紧急程度', ticket.urgency),
        el('p', { class: 'muted', style: 'font-size:12px;margin:10px 0 0' }, [
          `字段来源：${source} ｜ 填充度 ${filled} ｜ 工单台可继续跟进`,
        ]),
        el('a', { class: 'btn btn-primary btn-sm', href: '/admin', style: 'margin-top:10px', text: '前往工单台 →' }),
      ]),
    ]);
    const node = el('div', { class: 'msg assistant' }, [avatar('assistant'), el('div', { class: 'msg-body' }, [card])]);
    dom.messages.append(node);
    scrollToBottom();
  }
})();
