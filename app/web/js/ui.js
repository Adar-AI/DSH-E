/**
 * 通用 UI 工具：DOM 构造、时间格式化、状态与紧急程度渲染、轻提示。
 *
 * 安全约定：所有来自后端/用户的文本一律通过 textContent 写入，绝不拼 innerHTML，
 * 避免把模型输出或客户描述当成 HTML 执行。
 */
const UI = (() => {
  const STATUS_ORDER = ['待处理', '处理中', '已解决', '已关闭'];

  // 与后端 app/tickets/service.py 的 ALLOWED_TRANSITIONS 保持一致（后端仍是最终裁判）
  const ALLOWED_TRANSITIONS = {
    待处理: ['处理中', '已关闭'],
    处理中: ['已解决', '已关闭', '待处理'],
    已解决: ['已关闭', '处理中'],
    已关闭: ['处理中'],
  };

  const FIELD_LABELS = {
    customer_request: '客户诉求',
    product_model: '产品型号',
    fault_phenomenon: '故障现象',
    error_info: '报错信息',
    tried_actions: '已尝试操作',
    urgency: '紧急程度',
    suggestion: '处理建议',
  };

  /** 创建元素：UI.el('div', {class: 'x', text: '内容'}, [子元素...]) */
  function el(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(options).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      if (key === 'text') node.textContent = value;
      else if (key === 'value') node.value = value; // input / textarea / select 都适用
      else if (key === 'html') node.innerHTML = value; // 仅用于本文件内写死的图标/骨架
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key === 'class') node.className = value;
      else if (key.startsWith('on') && typeof value === 'function') {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else node.setAttribute(key, value);
    });
    (Array.isArray(children) ? children : [children]).forEach((child) => {
      if (!child) return;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function toDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatDateTime(value) {
    const date = toDate(value);
    if (!date) return '—';
    return date.toLocaleString('zh-CN', { hour12: false });
  }

  function formatShortTime(value) {
    const date = toDate(value);
    if (!date) return '—';
    const now = new Date();
    const sameDay = date.toDateString() === now.toDateString();
    const time = date.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' });
    if (sameDay) return time;
    return `${date.getMonth() + 1}-${String(date.getDate()).padStart(2, '0')} ${time}`;
  }

  function statusEl(status) {
    const text = status || '待处理';
    return el('span', { class: `status s-${text}`, text });
  }

  function urgencyBadge(urgency) {
    const text = urgency || '普通';
    const kind = text === '紧急' ? 'badge-danger' : text === '较急' ? 'badge-warning' : 'badge-muted';
    return el('span', { class: `badge ${kind}`, text });
  }

  function truncate(text, length = 80) {
    const value = String(text || '');
    return value.length > length ? `${value.slice(0, length)}…` : value;
  }

  /** 结构化字段用：去掉 markdown 记号、压成单行（工单字段是按行展示的）。 */
  function stripMarkdown(text) {
    return String(text || '')
      .replace(/\*\*(.+?)\*\*/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/^\s*[-*•]\s+/gm, '')
      .replace(/\s*\n\s*/g, ' ')
      .trim();
  }

  function appendInline(parent, text) {
    // 只处理 **加粗**：按 ** 切成若干段，奇数段加粗；其余一律当纯文本
    String(text)
      .split('**')
      .forEach((part, index) => {
        if (!part) return;
        if (index % 2 === 1) parent.append(el('strong', { text: part }));
        else parent.append(document.createTextNode(part));
      });
  }

  /**
   * 回答正文用的轻量富文本：支持 **加粗**、`- ` 列表、换行。
   * 全程用 DOM 节点拼装，不碰 innerHTML，因此模型输出无法注入脚本。
   */
  function richText(text) {
    const fragment = document.createDocumentFragment();
    String(text || '')
      .split('\n')
      .forEach((line) => {
        const trimmed = line.trim();
        if (!trimmed) {
          fragment.append(el('div', { class: 'md-gap' }));
          return;
        }
        const bullet = /^[-*•]\s+/.test(trimmed);
        const row = el('div', { class: bullet ? 'md-li' : 'md-p' });
        appendInline(row, bullet ? trimmed.replace(/^[-*•]\s+/, '') : trimmed);
        fragment.append(row);
      });
    return fragment;
  }

  /** 把富文本渲染进容器（清空后写入）。 */
  function renderRich(container, text) {
    clear(container).append(richText(text));
    return container;
  }

  function toast(message, kind = '') {
    let wrap = document.querySelector('.toast-wrap');
    if (!wrap) {
      wrap = el('div', { class: 'toast-wrap' });
      document.body.append(wrap);
    }
    const node = el('div', { class: `toast ${kind}`.trim(), text: message });
    wrap.append(node);
    setTimeout(() => node.remove(), kind === 'error' ? 5200 : 2800);
  }

  /** 「本轮依据」折叠块：来自 retrieval 事件的 chunks */
  function citationsBlock(chunks, options = {}) {
    const items = Array.isArray(chunks) ? chunks : [];
    if (!items.length) return null;

    const box = el('details', { class: 'citations', open: options.open === false ? null : '' }, [
      el('summary', { text: options.title || `本轮依据（${items.length} 条）` }),
    ]);

    items.forEach((chunk, index) => {
      const head = el('div', { class: 'cite-head' }, [
        el('span', { text: `[${index + 1}] ${chunk.citation || chunk.filename || '未知来源'}` }),
        el('span', { class: 'mono', text: (chunk.score ?? 0).toFixed(3) }),
      ]);
      const detail = el('div', { class: 'msg-meta mono' }, [
        `覆盖率 ${(chunk.coverage ?? 0).toFixed(2)} ｜ 向量 ${(chunk.vector_sim ?? 0).toFixed(2)} ｜ RRF ${(chunk.rrf ?? 0).toFixed(2)}`,
      ]);
      box.append(
        el('div', { class: 'cite-item' }, [
          head,
          el('div', { class: 'cite-text', text: truncate(chunk.text, 220) }),
          detail,
        ]),
      );
    });

    return box;
  }

  function fieldRow(label, value, options = {}) {
    const text = stripMarkdown(value);
    return el('div', { class: 'field-row' }, [
      el('span', { class: 'label', text: label }),
      text
        ? el('span', { class: 'value', text })
        : el('span', { class: 'value faint', text: options.empty || '（未填写）' }),
    ]);
  }

  return {
    STATUS_ORDER,
    ALLOWED_TRANSITIONS,
    FIELD_LABELS,
    el,
    clear,
    formatDateTime,
    formatShortTime,
    statusEl,
    urgencyBadge,
    truncate,
    stripMarkdown,
    richText,
    renderRich,
    toast,
    citationsBlock,
    fieldRow,
  };
})();
