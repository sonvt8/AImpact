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
      {loading && !data && <div className="panel empty-state" role="status">Đang tải thống kê…</div>}
      {data && <>
        <div className="metric-grid">
          {Object.entries(data.totals).map(([label, value]) => <div className="metric" key={label}><span>{label}</span><b>{value}</b></div>)}
        </div>
        {data.rows.length ? <div className="panel table-wrap">
          <table><thead><tr>{columns.map(column => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>{data.rows.map((row, index) => <tr key={index}>{columns.map(column => <td key={column}>{String(row[column] ?? '—')}</td>)}</tr>)}</tbody></table>
        </div> : <div className="panel empty-state">Sheet không có dữ liệu.</div>}
      </>}
    </div>
  )
}
