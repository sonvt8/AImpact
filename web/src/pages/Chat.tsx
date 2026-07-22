import { FormEvent, useEffect, useMemo, useState } from 'react'
import CitationPanel from '../components/CitationPanel'
import Markdown from '../components/Markdown'
import { api, authorizedFetch } from '../lib/api'
import type { Citation, Conversation, Message } from '../lib/types'

const NO_EVIDENCE = 'Không tìm thấy thông tin phù hợp trong tài liệu.'

export default function Chat() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [busy, setBusy] = useState(false)
  const [loadingConversations, setLoadingConversations] = useState(true)
  const [error, setError] = useState('')

  const citations = useMemo<Citation[]>(() => (
    [...messages].reverse().find(message => message.role === 'assistant' && message.citations?.length)?.citations || []
  ), [messages])

  async function loadConversations() {
    setLoadingConversations(true)
    try { setConversations(await api<Conversation[]>('/api/conversations')) }
    finally { setLoadingConversations(false) }
  }

  useEffect(() => { loadConversations().catch(reason => setError(reason.message)) }, [])

  async function openConversation(id: string) {
    const conversation = await api<Conversation>(`/api/conversations/${id}`)
    setConversationId(id)
    setMessages(conversation.messages || [])
  }

  async function newConversation() {
    setConversationId(null)
    setMessages([])
    setError('')
  }

  async function removeConversation(id: string) {
    await api(`/api/conversations/${id}`, { method: 'DELETE' })
    if (conversationId === id) await newConversation()
    await loadConversations()
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const question = query.trim()
    if (!question || busy) return
    setBusy(true)
    setError('')
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
      setMessages(previous => [...previous, { role: 'user', content: question }, { role: 'assistant', content: '', citations: [] }])
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
          setMessages(previous => previous.map((message, index) => {
            if (index !== previous.length - 1) return message
            if (event.type === 'token') return { ...message, content: message.content + event.text }
            if (event.type === 'final') return { ...message, content: event.text || message.content, citations: event.citations || [] }
            if (event.type === 'error') return { ...message, content: event.detail }
            return message
          }))
        }
        if (done) break
      }
      await loadConversations()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Truy vấn thất bại')
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
          {loadingConversations && !conversations.length ? <div className="conversation-empty" role="status">Đang tải hội thoại…</div> : !conversations.length ? <div className="conversation-empty">Chưa có hội thoại.</div> : conversations.map(conversation => (
            <div className={`conversation-item ${conversation.id === conversationId ? 'active' : ''}`} key={conversation.id}>
              <button onClick={() => openConversation(conversation.id)}>{conversation.title}</button>
              <button className="delete-mini" aria-label={`Xóa ${conversation.title}`} onClick={() => removeConversation(conversation.id)}>×</button>
            </div>
          ))}
        </div>
      </aside>
      <section className="chat-main" aria-busy={busy}>
        <div className="chat-status"><span className="status-dot" /><b>Evidence gate hoạt động</b><span>Ngưỡng quyết định do quản trị viên cấu hình</span></div>
        <div className="message-list">
          {!messages.length && (
            <div className="chat-empty">
              <span className="eyebrow">TRA CỨU CÓ KIỂM CHỨNG</span>
              <h1>Hỏi tài liệu vận hành</h1>
              <p>Đặt câu hỏi cụ thể. Hệ thống sẽ từ chối trả lời nếu không tìm thấy bằng chứng đạt ngưỡng.</p>
            </div>
          )}
          {messages.map((message, index) => (
            <article className={`message ${message.role} ${message.content === NO_EVIDENCE ? 'no-evidence' : ''}`} key={message.id || index}>
              <div className="message-role">{message.role === 'user' ? 'BẠN' : 'AIMPACT'}</div>
              {message.content ? <Markdown text={message.content} /> : <span className="typing" role="status">Đang tổng hợp từ bằng chứng…</span>}
            </article>
          ))}
          {error && <div className="error-box" role="alert">{error}</div>}
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
    </div>
  )
}
