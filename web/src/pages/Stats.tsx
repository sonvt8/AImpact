import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { notify } from '../notify'

type StatsResult = { sheet: string; rows: Record<string, unknown>[]; totals: Record<string, number> }

export default function Stats() {
  const [sheet, setSheet] = useState('Tong hop')
  const [data, setData] = useState<StatsResult | null>(null)
  const [loading, setLoading] = useState(true)
  const columns = useMemo(() => data?.rows[0] ? Object.keys(data.rows[0]) : [], [data])

  async function load() {
    setLoading(true)
    try { setData(await api<StatsResult>(`/api/stats?sheet=${encodeURIComponent(sheet)}`)) }
    catch (reason) { notify.error('Không đọc được thống kê', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
    finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])

  return (
    <div className="page" aria-busy={loading}>
      <div className="page-heading">
        <div><h1>Thống kê sự cố</h1><p>Tính trực tiếp từ bảng tính, không qua LLM.</p></div>
        <div className="sheet-picker"><input aria-label="Tên sheet" disabled={loading} value={sheet} onChange={event => setSheet(event.target.value)} /><button className="button secondary" disabled={loading} onClick={load}>{loading ? 'Đang đọc…' : 'Đọc sheet'}</button></div>
      </div>
      {loading && !data && <div role="status" aria-label="Đang tải thống kê">
        <span className="sr-only">Đang tải thống kê…</span>
        <div className="metric-grid" aria-hidden="true">
          {[0, 1, 2, 3].map(item => <div className="metric skeleton skeleton-metric" key={item} />)}
        </div>
        <div className="panel skeleton-table" aria-hidden="true">
          {[0, 1, 2, 3, 4].map(item => <div className="skeleton skeleton-line" key={item} />)}
        </div>
      </div>}
      {data && <>
        <div className="metric-grid">
          {Object.entries(data.totals).map(([label, value]) => <div className="metric" key={label}><span>{label}</span><b>{value}</b></div>)}
        </div>
        {data.rows.length ? <div className="panel table-wrap">
          <table><thead><tr>{columns.map(column => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>{data.rows.map((row, index) => <tr key={index}>{columns.map(column => <td key={column}>{String(row[column] ?? '—')}</td>)}</tr>)}</tbody></table>
        </div> : <div className="panel empty-state empty-state-rich">
          <svg className="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M4 19V9M10 19V5M16 19v-7M22 19H2" />
          </svg>
          <strong className="empty-state-title">Sheet chưa có dữ liệu</strong>
          <p className="empty-state-guidance">Nhập tên sheet có dữ liệu rồi chọn “Đọc sheet”.</p>
        </div>}
      </>}
    </div>
  )
}
