/**
 * 人工客服工单台逻辑（app/web/admin.html）。
 *
 * 功能：列表（状态筛选 / 关键字搜索 / 统计）、详情抽屉、人工修正 AI 抽取字段、
 * 底部固定的状态流转操作栏、CSV 导出。
 * 状态流转的合法性由后端裁决（app/tickets/service.py），前端只按同一张表渲染可用按钮。
 */
(() => {
  const el = UI.el;

  const state = {
    status: '',
    q: '',
    tickets: [],
    stats: {},
    selected: null,
    editing: false,
    saving: false,
  };

  const dom = {};

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    dom.tabs = document.getElementById('tabs');
    dom.search = document.getElementById('search');
    dom.refresh = document.getElementById('btn-refresh');
    dom.export = document.getElementById('btn-export');
    dom.count = document.getElementById('ticket-count');
    dom.rows = document.getElementById('ticket-rows');
    dom.drawer = document.getElementById('drawer');

    dom.refresh.addEventListener('click', () => loadTickets(true));
    dom.export.addEventListener('click', () => {
      window.location.href = API.exportUrl({ status: state.status, q: state.q });
    });
    dom.search.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        state.q = dom.search.value.trim();
        loadTickets();
      }
    });
    let timer = null;
    dom.search.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        state.q = dom.search.value.trim();
        loadTickets();
      }, 350);
    });

    renderDrawerEmpty();
    loadTickets().then(() => {
      // 支持深链 /admin?ticket=<id>，便于把某张工单直接发给同事
      const requested = new URLSearchParams(window.location.search).get('ticket');
      if (requested) openTicket(Number(requested));
    });
  }

  /* ------------------------------------------------------------ 列表 */

  async function loadTickets(showToast = false) {
    try {
      const data = await API.tickets({ status: state.status, q: state.q, limit: 200 });
      state.tickets = data.items || [];
      state.stats = data.stats || {};
      dom.count.textContent = `共 ${data.total} 张工单`;
      renderTabs();
      renderRows();
      if (showToast) UI.toast('已刷新');
      if (state.selected) {
        const stillThere = state.tickets.some((item) => item.id === state.selected.id);
        if (stillThere) await openTicket(state.selected.id, true);
        else {
          state.selected = null;
          renderDrawerEmpty();
        }
      }
    } catch (error) {
      UI.toast(error.message, 'error');
    }
  }

  function renderTabs() {
    UI.clear(dom.tabs);
    const entries = [['', '全部'], ...UI.STATUS_ORDER.map((status) => [status, status])];
    entries.forEach(([value, label]) => {
      const count = value === '' ? state.stats['全部'] ?? 0 : state.stats[value] ?? 0;
      dom.tabs.append(
        el('button', {
          type: 'button',
          class: value === state.status ? 'active' : '',
          text: `${label} ${count}`,
          onclick: () => {
            state.status = value;
            loadTickets();
          },
        }),
      );
    });
  }

  function renderRows() {
    UI.clear(dom.rows);
    if (!state.tickets.length) {
      dom.rows.append(
        el('tr', {}, [
          el('td', { colspan: '7' }, [
            el('div', { class: 'loading', text: state.q ? '没有匹配的工单' : '还没有工单，先在对话页结束一次会话试试' }),
          ]),
        ]),
      );
      return;
    }

    state.tickets.forEach((ticket) => {
      const row = el('tr', { class: state.selected && state.selected.id === ticket.id ? 'active' : '' }, [
        el('td', { class: 'mono', text: ticket.ticket_no }),
        el('td', {}, [UI.statusEl(ticket.status)]),
        el('td', {}, [UI.urgencyBadge(ticket.urgency)]),
        el('td', { class: 'mono', text: ticket.product_model || '—' }),
        el('td', { class: 'mono', text: ticket.error_info || '—' }),
        el('td', {}, [el('div', { class: 'cell-clamp', text: ticket.customer_request || '—' })]),
        el('td', { class: 'muted', text: UI.formatShortTime(ticket.updated_at) }),
      ]);
      row.addEventListener('click', () => openTicket(ticket.id));
      dom.rows.append(row);
    });
  }

  /* ------------------------------------------------------------ 详情 */

  async function openTicket(id, silent = false) {
    try {
      const ticket = await API.ticket(id);
      state.selected = ticket;
      state.editing = false;
      renderRows();
      renderDrawer(ticket);
      if (!silent) dom.drawer.scrollTop = 0;
    } catch (error) {
      UI.toast(error.message, 'error');
    }
  }

  function renderDrawerEmpty() {
    UI.clear(dom.drawer);
    dom.drawer.append(
      el('div', { class: 'empty-state' }, [
        el('div', { class: 'icon', text: '📋' }),
        el('div', { text: '从左侧选择一张工单查看详情' }),
        el('div', { class: 'muted', style: 'font-size:12px', text: '详情包含 AI 定稿字段、资料来源、状态流转与处理历史' }),
      ]),
    );
  }

  function renderDrawer(ticket) {
    UI.clear(dom.drawer);

    const head = el('div', { class: 'drawer-head' }, [
      el('div', {}, [
        el('strong', { class: 'mono', text: ticket.ticket_no }),
        el('span', { style: 'margin-left:8px' }, [UI.statusEl(ticket.status)]),
      ]),
      el('button', {
        class: 'btn btn-ghost btn-sm',
        type: 'button',
        text: '✕',
        onclick: () => {
          state.selected = null;
          state.editing = false;
          renderRows();
          renderDrawerEmpty();
        },
      }),
    ]);

    const sections = el('div', { id: 'drawer-sections' });
    sections.append(fieldsSection(ticket));
    sections.append(citationsSection(ticket));
    sections.append(eventsSection(ticket));

    dom.drawer.append(head, sections, actionsBar(ticket));
  }

  /* ---- 字段区（含人工编辑） ---- */

  const MULTILINE = new Set(['customer_request', 'fault_phenomenon', 'suggestion']);

  function fieldsSection(ticket) {
    const section = el('div', { class: 'section' });
    const head = el('div', { class: 'section-head' }, [
      el('h4', { text: state.editing ? '编辑工单字段' : 'AI 定稿字段' }),
      state.editing
        ? null
        : el('button', {
            class: 'btn btn-sm',
            type: 'button',
            text: '✎ 编辑字段',
            onclick: () => {
              state.editing = true;
              renderDrawer(state.selected);
            },
          }),
    ]);
    section.append(head);

    if (!state.editing) {
      ['customer_request', 'product_model', 'fault_phenomenon', 'error_info', 'tried_actions'].forEach((key) => {
        section.append(UI.fieldRow(UI.FIELD_LABELS[key], ticket[key]));
      });
      section.append(
        el('div', { class: 'field-row' }, [
          el('span', { class: 'label', text: '紧急程度' }),
          el('span', { class: 'value' }, [UI.urgencyBadge(ticket.urgency)]),
        ]),
      );
      section.append(UI.fieldRow('处理建议', ticket.suggestion));
      const extraction = ticket.extraction;
      if (extraction && extraction.source) {
        const source = extraction.source === 'llm' ? 'AI 结构化抽取' : extraction.source === 'existing' ? '已有工单' : '规则抽取';
        const filled = Array.isArray(extraction.filled) ? `${extraction.filled[0]}/${extraction.filled[1]}` : '—';
        section.append(
          el('p', { class: 'muted', style: 'font-size:12px;margin:8px 0 0' }, [
            `字段来源：${source} ｜ 核心字段填充度 ${filled} ｜ 创建于 ${UI.formatDateTime(ticket.created_at)}`,
          ]),
        );
      }
      if (ticket.conv_id) {
        section.append(
          el('p', { style: 'font-size:12px;margin:6px 0 0' }, [
            el('span', { class: 'muted', text: '来源会话：' }),
            el('a', {
              class: 'mono',
              href: `/?conv=${encodeURIComponent(ticket.conv_id)}`,
              text: String(ticket.conv_id).slice(0, 18),
            }),
          ]),
        );
      }
      return section;
    }

    const inputs = {};
    ['customer_request', 'product_model', 'fault_phenomenon', 'error_info', 'tried_actions'].forEach((key) => {
      const input = el(MULTILINE.has(key) ? 'textarea' : 'input', {
        value: ticket[key] || '',
        placeholder: `填写${UI.FIELD_LABELS[key]}`,
      });
      if (key === 'product_model' || key === 'error_info') input.setAttribute('maxlength', '120');
      inputs[key] = input;
      section.append(el('div', { class: 'field-row' }, [el('span', { class: 'label', text: UI.FIELD_LABELS[key] }), input]));
    });

    const urgencySelect = el('select', {});
    ['普通', '较急', '紧急'].forEach((value) => {
      const option = el('option', { value, text: value });
      if ((ticket.urgency || '普通') === value) option.selected = true;
      urgencySelect.append(option);
    });
    inputs.urgency = urgencySelect;
    section.append(el('div', { class: 'field-row' }, [el('span', { class: 'label', text: '紧急程度' }), urgencySelect]));

    const suggestion = el('textarea', { value: ticket.suggestion || '', placeholder: '填写处理建议' });
    inputs.suggestion = suggestion;
    section.append(el('div', { class: 'field-row' }, [el('span', { class: 'label', text: '处理建议' }), suggestion]));

    section.append(
      el('div', { class: 'edit-actions' }, [
        el('button', {
          class: 'btn btn-primary btn-sm',
          type: 'button',
          text: state.saving ? '保存中…' : '保存修改',
          disabled: state.saving ? '' : null,
          onclick: () => saveFields(inputs),
        }),
        el('button', {
          class: 'btn btn-sm',
          type: 'button',
          text: '取消',
          onclick: () => {
            state.editing = false;
            renderDrawer(state.selected);
          },
        }),
      ]),
    );
    section.append(
      el('p', { class: 'muted', style: 'font-size:12px;margin:8px 0 0', text: '修改会记入下方处理历史，工单状态不受影响。' }),
    );
    return section;
  }

  async function saveFields(inputs) {
    const fields = {};
    Object.entries(inputs).forEach(([key, input]) => {
      fields[key] = input.value.trim();
    });
    state.saving = true;
    renderDrawer(state.selected);
    try {
      const updated = await API.updateTicketFields(state.selected.id, fields);
      state.selected = updated;
      state.editing = false;
      renderDrawer(updated);
      renderRows();
      UI.toast('字段已更新', 'success');
      loadTickets();
    } catch (error) {
      UI.toast(error.message, 'error');
    } finally {
      state.saving = false;
    }
  }

  /* ---- 依据 ---- */

  function citationsSection(ticket) {
    const section = el('div', { class: 'section' }, [el('h4', { text: '资料来源' })]);
    const citations = Array.isArray(ticket.citations) ? ticket.citations : [];
    if (!citations.length) {
      section.append(el('p', { class: 'muted', style: 'font-size:13px', text: '该工单没有关联资料（可能是未命中知识库的兜底会话）。' }));
      return section;
    }
    citations.forEach((citation, index) => {
      section.append(
        el('div', { class: 'cite-item' }, [
          el('div', { class: 'cite-head' }, [el('span', { text: `[${index + 1}] ${citation}` })]),
        ]),
      );
    });
    return section;
  }

  /* ---- 处理历史 ---- */

  function eventsSection(ticket) {
    const section = el('div', { class: 'section' }, [el('h4', { text: '处理历史' })]);
    const events = Array.isArray(ticket.events) ? ticket.events : [];
    if (!events.length) {
      section.append(el('p', { class: 'muted', style: 'font-size:13px', text: '暂无记录' }));
      return section;
    }
    const list = el('ul', { class: 'timeline' });
    events.forEach((event) => {
      const title =
        event.from_status && event.from_status !== event.to_status
          ? `${event.from_status} → ${event.to_status}`
          : event.note || '记录';
      const lines = [];
      if (event.from_status && event.from_status !== event.to_status) {
        if (event.note) lines.push(event.note);
      }
      list.append(
        el('li', {}, [
          el('div', { text: lines.length ? `${title}：${lines[0]}` : title }),
          el('div', { class: 'when', text: UI.formatDateTime(event.created_at) }),
        ]),
      );
    });
    section.append(list);
    return section;
  }

  /* ---- 底部固定操作栏 ---- */

  function actionsBar(ticket) {
    const allowed = UI.ALLOWED_TRANSITIONS[ticket.status] || [];
    const note = el('textarea', {
      class: 'textarea',
      rows: '2',
      placeholder: '填写处理备注（可选），例如：已电话联系客户确认型号',
    });

    const buttons = allowed.map((target) =>
      el('button', {
        class: target === '已解决' ? 'btn btn-primary btn-sm' : 'btn btn-sm',
        type: 'button',
        text: `→ ${target}`,
        onclick: () => transition(target, note.value.trim()),
      }),
    );

    if (!buttons.length) {
      buttons.push(el('span', { class: 'muted', style: 'font-size:12px', text: '当前状态没有可用的流转' }));
    }

    return el('div', { class: 'sticky-actions' }, [
      note,
      el('div', { class: 'actions', style: 'margin-top:10px' }, buttons),
      el('p', { class: 'muted', style: 'font-size:12px;margin:8px 0 0' }, [
        '每次流转都会写入处理历史；不合法的流转会被后端拒绝。',
      ]),
    ]);
  }

  async function transition(target, note) {
    if (!state.selected) return;
    try {
      const updated = await API.updateTicketStatus(state.selected.id, target, note);
      state.selected = updated;
      state.editing = false;
      renderDrawer(updated);
      renderRows();
      UI.toast(`已变更为「${target}」`, 'success');
      loadTickets();
    } catch (error) {
      UI.toast(error.message, 'error');
    }
  }
})();
