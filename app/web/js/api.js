/**
 * 后端接口封装：所有网络调用集中在这里，页面脚本只关心数据与渲染。
 *
 * 说明：对话接口是 POST multipart，浏览器原生 EventSource 只支持 GET，
 * 所以 SSE 用 fetch + ReadableStream 手动解析（与后端 app/api/chat.py 的事件格式一致）。
 */
const API = (() => {
  async function request(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = `请求失败（HTTP ${response.status}）`;
      try {
        const payload = await response.json();
        if (payload && payload.detail) {
          detail = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail);
        }
      } catch (_) {
        /* 响应不是 JSON，用默认文案 */
      }
      throw new Error(detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function jsonOptions(method, body) {
    return {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    };
  }

  function buildQuery(params) {
    const search = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') search.append(key, value);
    });
    const text = search.toString();
    return text ? `?${text}` : '';
  }

  function parseBlock(block) {
    let name = 'message';
    const dataLines = [];
    for (const line of block.split('\n')) {
      const text = line.replace(/\r$/, '');
      if (text.startsWith('event:')) name = text.slice(6).trim();
      else if (text.startsWith('data:')) dataLines.push(text.slice(5).trim());
    }
    if (!dataLines.length) return null;
    let data;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch (_) {
      data = { raw: dataLines.join('\n') };
    }
    return { name, data };
  }

  /** 发送一轮对话，按事件回调流式消费（meta / stage / vision / retrieval / token / draft / done / error）。 */
  async function streamChat({ text, conversationId, image, onEvent, signal }) {
    const form = new FormData();
    form.append('text', text || '');
    if (conversationId) form.append('conversation_id', conversationId);
    if (image) form.append('image', image);

    const response = await fetch('/api/chat', { method: 'POST', body: form, signal });
    if (!response.ok) {
      let detail = `请求失败（HTTP ${response.status}）`;
      try {
        const payload = await response.json();
        if (payload && payload.detail) detail = payload.detail;
      } catch (_) {
        /* 忽略 */
      }
      throw new Error(detail);
    }
    if (!response.body) throw new Error('当前浏览器不支持流式响应');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index = buffer.indexOf('\n\n');
      while (index >= 0) {
        const event = parseBlock(buffer.slice(0, index));
        buffer = buffer.slice(index + 2);
        if (event) onEvent(event);
        index = buffer.indexOf('\n\n');
      }
    }
    const tail = parseBlock(buffer);
    if (tail) onEvent(tail);
  }

  return {
    health: () => request('/api/health'),

    conversations: (limit = 50) => request(`/api/conversations${buildQuery({ limit })}`),
    conversation: (id) => request(`/api/conversations/${encodeURIComponent(id)}`),
    messages: (id, limit) =>
      request(`/api/conversations/${encodeURIComponent(id)}/messages${buildQuery({ limit })}`),
    messagesAfter: (id, afterId) =>
      request(
        `/api/conversations/${encodeURIComponent(id)}/messages${buildQuery({ after_id: afterId })}`,
      ),
    requestHandoff: (id, reason = '') =>
      request(`/api/conversations/${encodeURIComponent(id)}/handoff${buildQuery({ reason })}`, {
        method: 'POST',
      }),
    handoffs: () => request('/api/handoffs'),
    agentReply: (id, text) =>
      request(
        `/api/conversations/${encodeURIComponent(id)}/agent-messages${buildQuery({ text })}`,
        { method: 'POST' },
      ),
    deleteConversation: (id) =>
      request(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    closeConversation: (id, force = false) =>
      request(`/api/conversations/${encodeURIComponent(id)}/close${buildQuery({ force })}`, {
        method: 'POST',
      }),

    tickets: ({ status, q, convId, limit = 50, offset = 0 } = {}) =>
      request(`/api/tickets${buildQuery({ status, q, conv_id: convId, limit, offset })}`),
    ticket: (id) => request(`/api/tickets/${id}`),
    updateTicketStatus: (id, toStatus, note = '', operator = '客服') =>
      request(
        `/api/tickets/${id}`,
        jsonOptions('PATCH', { to_status: toStatus, note, operator }),
      ),
    updateTicketFields: (id, fields, note = '', operator = '客服') =>
      request(
        `/api/tickets/${id}/fields`,
        jsonOptions('PATCH', { fields, note, operator }),
      ),
    exportUrl: ({ status, q } = {}) => `/api/tickets/export.csv${buildQuery({ status, q })}`,

    streamChat,
  };
})();
