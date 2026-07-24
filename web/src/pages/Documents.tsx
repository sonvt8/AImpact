import { ChangeEvent, DragEvent, useEffect, useState } from 'react'
import ConfirmDialog from '../components/ConfirmDialog'
import { api } from '../lib/api'
import { notify } from '../notify'

export default function Documents() {
  const [documents, setDocuments] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pendingFilename, setPendingFilename] = useState<string | null>(null)

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

  function confirmRemove() {
    const filename = pendingFilename
    setPendingFilename(null)
    if (filename) void remove(filename)
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
          {loading && !documents.length ? <div className="document-list skeleton-list" role="status" aria-label="Đang tải tài liệu">{[0, 1, 2].map(row => <div className="skeleton-row" aria-hidden="true" key={row}><span className="skeleton skeleton-text" /><span className="skeleton skeleton-action" /></div>)}</div> : !documents.length ? <div className="empty-state empty-state-rich">
            <svg className="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8m-6-6 6 6m-6-6v6h6M8 13h8m-8 4h5" /></svg>
            <b className="empty-state-title">Chưa có tài liệu</b>
            <span className="empty-state-guidance">Kéo thả tệp XLSX, PDF, DOCX, TXT hoặc CSV để bắt đầu lập chỉ mục.</span>
          </div> : (
            <div className="document-list">
              {documents.map(filename => <div key={filename}><span className="mono">{filename}</span><button className="button danger" onClick={() => setPendingFilename(filename)}>Xóa</button></div>)}
            </div>
          )}
        </section>
      </div>
      <ConfirmDialog open={pendingFilename !== null} title="Xóa tài liệu?" description={`Tài liệu “${pendingFilename ?? ''}” sẽ bị xóa khỏi chỉ mục. Hành động này không thể hoàn tác.`} onCancel={() => setPendingFilename(null)} onConfirm={confirmRemove} />
    </div>
  )
}
