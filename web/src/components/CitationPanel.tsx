import { authorizedFetch } from '../lib/api'
import type { Citation } from '../lib/types'

async function openSource(filename: string) {
  const response = await authorizedFetch(`/api/documents/${encodeURIComponent(filename)}`)
  if (!response.ok) return
  const url = URL.createObjectURL(await response.blob())
  window.open(url, '_blank', 'noopener,noreferrer')
  setTimeout(() => URL.revokeObjectURL(url), 60_000)
}

export default function CitationPanel({ citations }: { citations: Citation[] }) {
  return (
    <aside className="citation-panel" aria-label="Bằng chứng nguồn">
      <div className="panel-heading">
        <span className="status-dot" aria-hidden="true" />
        <div><small>ĐƯỜNG DẪN BẰNG CHỨNG</small><strong>{citations.length} nguồn xác thực</strong></div>
      </div>
      {!citations.length && (
        <div className="empty-state compact">
          Trích dẫn nguyên văn sẽ xuất hiện tại đây khi câu trả lời có đủ bằng chứng.
        </div>
      )}
      <div className="citation-list">
        {citations.map((citation, index) => {
          const score = Math.round(citation.similarity * 100)
          return (
            <article className="citation-card" key={`${citation.locator}-${index}`}>
              <button className="citation-link" type="button" onClick={() => openSource(citation.filename)}>
                <span>{citation.filename}</span>
                <span>{citation.sheet_name || '—'} · {citation.locator}</span>
              </button>
              <div className="meter-row">
                <span>TƯƠNG ĐỒNG</span><b>{score}%</b>
                <div className="meter" role="meter" aria-label={`Tương đồng ${score}%`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={score}><i style={{ width: `${score}%` }} /></div>
              </div>
              {citation.section_path && <div className="section-path">{citation.section_path}</div>}
              <blockquote>{citation.content}</blockquote>
            </article>
          )
        })}
      </div>
    </aside>
  )
}
