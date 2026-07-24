import { FormEvent, useEffect, useMemo, useState } from 'react'
import CitationPanel from '../components/CitationPanel'
import ConfirmDialog from '../components/ConfirmDialog'
import Markdown from '../components/Markdown'
import { api, authorizedFetch } from '../lib/api'
import type { Citation, Conversation, Message } from '../lib/types'
import { notify } from '../notify'

const NO_EVIDENCE = 'Không tìm thấy thông tin phù hợp trong tài liệu.'

type ChatMessage = Message & { route?: 'structured' | 'retrieval' | 'refuse', stageLabel?: string }

export default function Chat() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [busy, setBusy] = useState(false)
  const [loadingConversations, setLoadingConversations] = useState(true)
  const [pendingConversation, setPendingConversation] = useState<Conversation | null>(null)

  const citations = useMemo<Citation[]>(() => (
    [...messages].reverse().find(message => message.role === 'assistant' && message.citations?.length)?.citations || []
  ), [messages])

  async function loadConversations() {
    setLoadingConversations(true)
    try { setConversations(await api<Conversation[]>('/api/conversations')) }
    finally { setLoadingConversations(false) }
  }

  useEffect(() => { loadConversations().catch(reason => notify.error(reason instanceof Error ? reason.message : 'Không tải được hội thoại')) }, [])

  async function openConversation(id: string) {
    try {
      const conversation = await api<Conversation>(`/api/conversations/${id}`)
      setConversationId(id)
      setMessages(conversation.messages || [])
    } catch (reason) {
      notify.error('Không mở được hội thoại', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' })
    }
  }

  async function newConversation() {
    setConversationId(null)
    setMessages([])
  }

  async function removeConversation(id: string) {
    try {
      await api(`/api/conversations/${id}`, { method: 'DELETE' })
      if (conversationId === id) await newConversation()
      await loadConversations()
      notify.success('Đã xóa hội thoại')
    } catch (reason) {
      notify.error('Không thể xóa hội thoại', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' })
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const question = query.trim()
    if (!question || busy) return
    setBusy(true)
    setQuery('')
    try {
      let activeId = conversationId
      if (!activeId) {
        const created = await api<Conversation>('/api/conversations', {
          method: 'POST',
          body: JSON.stringify({ title: question.slice(0, 80) }),
        })
        activeId = created.id
        setConversationId(activeId)
      }
      setMessages(previous => [...previous, { role: 'user', content: question }, { role: 'assistant', content: '', citations: [], stageLabel: 'Đang chuẩn bị truy vấn' }])
      const response = await authorizedFetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, conversation_id: activeId, top_k: topK }),
      })
      if (!response.ok || !response.body) {
        const detail = await response.json().catch(() => ({ detail: 'Không thể gửi truy vấn' }))
        throw new Error(detail.detail)
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''
        for (const block of blocks) {
          const line = block.split('\n').find(item => item.startsWith('data:'))
          if (!line) continue
          const event = JSON.parse(line.slice(5).trim())
          if (event.type === 'error') {
            notify.error('Truy vấn thất bại', { description: event.detail || 'Không thể hoàn tất câu trả lời' })
            setMessages(previous => previous.slice(0, -1))
            continue
          }
          setMessages(previous => previous.map((message, index) => {
            if (index !== previous.length - 1) return message
            if (event.type === 'stage') return { ...message, stageLabel: event.label || message.stageLabel }
            if (event.type === 'token') return { ...message, content: message.content + event.text }
            if (event.type === 'final') return { ...message, content: event.text || message.content, citations: event.citations || [], route: event.route }
            return message
          }))
        }
        if (done) break
      }
      await loadConversations()
    } catch (reason) {
      setMessages(previous => {
        const last = previous[previous.length - 1]
        return last?.role === 'assistant' && !last.content ? previous.slice(0, -1) : previous
      })
      notify.error('Truy vấn thất bại', { description: reason instanceof Error ? reason.message : 'Không thể gửi truy vấn' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="chat-layout">
      <aside className="conversation-sidebar">
        <button className="button new-chat" onClick={newConversation}>＋ Hội thoại mới</button>
        <div className="conversation-label">HỘI THOẠI CỦA BẠN</div>
        <div className="conversation-list" aria-busy={loadingConversations}>
          {loadingConversations && !conversations.length ? <div className="conversation-empty" role="status">Đang tải hội thoại…</div> : !conversations.length ? (
            <div className="conversation-empty empty-state-rich compact">
              <svg className="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                <path d="M7 18.5 3.5 21v-5A8.5 8.5 0 1 1 7 18.5Z" />
                <path d="M8 9h8M8 13h5" />
              </svg>
              <strong className="empty-state-title">Chưa có hội thoại</strong>
              <p className="empty-state-guidance">Chọn “Hội thoại mới” để bắt đầu tra cứu tài liệu.</p>
            </div>
          ) : conversations.map(conversation => (
            <div className={`conversation-item ${conversation.id === conversationId ? 'active' : ''}`} key={conversation.id}>
              <button type="button" onClick={() => openConversation(conversation.id)}>{conversation.title}</button>
              <button type="button" className="delete-mini" aria-label={`Xóa ${conversation.title}`} onClick={() => setPendingConversation(conversation)}>×</button>
            </div>
          ))}
        </div>
      </aside>
      <section className="chat-main" aria-busy={busy}>
        <div className="chat-status"><span className="status-dot" /><b>Evidence gate hoạt động</b><span>Ngưỡng quyết định do quản trị viên cấu hình</span></div>
        <div className="message-list">
          {!messages.length && (
            <div className="chat-empty empty-state-rich">
              <svg className="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <path d="M14.5 4.5a6 6 0 1 0 0 12 6 6 0 0 0 0-12Z" />
                <path d="m19 19-2.8-2.8M5 5v4M3 7h4" />
              </svg>
              <span className="eyebrow">TRA CỨU CÓ KIỂM CHỨNG</span>
              <h1 className="empty-state-title">Hỏi tài liệu vận hành</h1>
              <p className="empty-state-guidance">Đặt câu hỏi cụ thể; hệ thống chỉ trả lời khi tìm thấy bằng chứng đạt ngưỡng.</p>
            </div>
          )}
          {messages.map((message, index) => (
            <article className={`message ${message.role} ${message.content === NO_EVIDENCE ? 'no-evidence' : ''}`} key={message.id || index}>
              <div className="message-role">{message.role === 'user' ? 'BẠN' : 'AIMPACT'}</div>
              {message.role === 'assistant' && (message.route || message.citations?.length) ? (
                <div className="message-meta">
                  {message.route && <span className="route-badge" data-route={message.route} aria-label={`Tuyến xử lý: ${message.route}`}>{message.route}</span>}
                  {message.citations?.map((citation, citationIndex) => (
                    <span className="source-badge" key={`${citation.filename}:${citation.locator}:${citationIndex}`} title={`${citation.filename} · ${citation.locator}`}>
                      Nguồn {citationIndex + 1} · {citation.filename} · {citation.locator}
                    </span>
                  ))}
                </div>
              ) : null}
              {message.content ? <Markdown text={message.content} /> : (
                <span className="stage-indicator" role="status" aria-live="polite">
                  <span className="stage-spinner" aria-hidden="true" />
                  <span className="stage-label" key={message.stageLabel}>{message.stageLabel || 'Đang xử lý yêu cầu'}</span>
                </span>
              )}
            </article>
          ))}
        </div>
        <form className="composer" onSubmit={submit}>
          <textarea aria-label="Câu hỏi" rows={2} placeholder="Ví dụ: Khi mất một lộ điện lưới tại N6 cần xử lý thế nào?" value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() }
          }} />
          <div className="composer-actions">
            <label>TOP K <input type="number" min="1" max="50" value={topK} onChange={event => setTopK(Number(event.target.value))} /></label>
            <span>Enter để gửi · Shift+Enter xuống dòng</span>
            <button className="button" disabled={busy || !query.trim()}>{busy ? 'Đang tra cứu…' : 'Gửi câu hỏi'}</button>
          </div>
        </form>
      </section>
      <CitationPanel citations={citations} />
      <ConfirmDialog
        open={Boolean(pendingConversation)}
        title="Xóa hội thoại?"
        description={pendingConversation ? `Hội thoại “${pendingConversation.title}” sẽ bị xóa vĩnh viễn.` : ''}
        onCancel={() => setPendingConversation(null)}
        onConfirm={() => {
          if (!pendingConversation) return
          const { id } = pendingConversation
          setPendingConversation(null)
          void removeConversation(id)
        }}
      />
    </div>
  )
}
