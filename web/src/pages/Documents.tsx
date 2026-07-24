import { ChangeEvent, DragEvent, useEffect, useState } from 'react'
import { api } from '../lib/api'
import { notify } from '../notify'

export default function Documents() {
  const [documents, setDocuments] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    try { setDocuments(await api<string[]>('/api/documents')) }
    finally { setLoading(false) }
  }
  useEffect(() => { load().catch(reason => notify.error(reason instanceof Error ? reason.message : 'Không tải được danh sách tài liệu')) }, [])

  async function upload(file?: File) {
    if (!file) return
    setBusy(true)
    const body = new FormData()
    body.append('file', file)
    try {
      const result = await api<{ status: string; added: number }>('/api/documents', { method: 'POST', body })
      notify.success(`Đã ingest ${file.name}`, { description: `${result.status}, thêm ${result.added} bản ghi.` })
      await load()
    } catch (reason) {
      notify.error('Không thể ingest tài liệu', { description: reason instanceof Error ? reason.message : 'Upload thất bại' })
    } finally {
      setBusy(false)
    }
  }

  async function remove(filename: string) {
    try {
      await api(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      await load()
      notify.success(`Đã xóa ${filename}`)
    } catch (reason) {
      notify.error('Không thể xóa tài liệu', { description: reason instanceof Error ? reason.message : 'Xóa tài liệu thất bại' })
    }
  }

  function drop(event: DragEvent) {
    event.preventDefault()
    upload(event.dataTransfer.files[0])
  }

  return (
    <div className="page">
      <div className="page-heading"><div><h1>Kho tài liệu</h1><p>Ingest dùng trực tiếp pipeline parser đã hardened của lõi RAG.</p></div></div>
      <div className="document-grid">
        <label className={`drop-zone ${busy ? 'busy' : ''}`} aria-busy={busy} onDrop={drop} onDragOver={event => event.preventDefault()}>
          <input type="file" accept=".xlsx,.pdf,.docx,.txt,.csv" disabled={busy} onChange={(event: ChangeEvent<HTMLInputElement>) => upload(event.target.files?.[0])} />
          <b>{busy ? 'Đang ingest tài liệu…' : 'Thả tệp vào đây'}</b>
          <span>hoặc bấm để chọn · XLSX, PDF, DOCX, TXT, CSV</span>
        </label>
        <section className="panel" aria-busy={loading}>
          <div className="panel-title"><b>Tài liệu đã lập chỉ mục</b><span>{documents.length} tệp</span></div>
          {loading && !documents.length ? <div className="empty-state" role="status">Đang tải tài liệu…</div> : !documents.length ? <div className="empty-state">Chưa có tài liệu trong chỉ mục.</div> : (
            <div className="document-list">
              {documents.map(filename => <div key={filename}><span className="mono">{filename}</span><button className="button danger" onClick={() => remove(filename)}>Xóa</button></div>)}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
