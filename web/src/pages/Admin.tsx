import { FormEvent, useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { Provider, Role } from '../lib/types'
import { notify } from '../notify'

type UserRow = { id: number; username: string; role: Role; created_at: string }
type AuditRow = { id: number; username: string; action: string; resource_id?: string; timestamp: string }

export default function Admin() {
  const [providers, setProviders] = useState<Provider[]>([])
  const [users, setUsers] = useState<UserRow[]>([])
  const [audit, setAudit] = useState<AuditRow[]>([])
  const [threshold, setThreshold] = useState(.78)
  const [models, setModels] = useState<Record<string, string[]>>({})
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<Role>('user')
  const [loading, setLoading] = useState(true)
  const [loadingModels, setLoadingModels] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    try {
      const [providerRows, userRows, thresholdResult, auditRows] = await Promise.all([
        api<Provider[]>('/api/providers'), api<UserRow[]>('/api/users'),
        api<{ threshold: number }>('/api/settings/threshold'), api<AuditRow[]>('/api/audit'),
      ])
      setProviders(providerRows); setUsers(userRows); setThreshold(thresholdResult.threshold); setAudit(auditRows)
    } finally { setLoading(false) }
  }
  useEffect(() => { load().catch(reason => notify.error(reason instanceof Error ? reason.message : 'Không tải được trang quản trị')) }, [])

  async function activate(id: string) {
    try {
      await api('/api/providers/active', { method: 'POST', body: JSON.stringify({ id }) })
      await load()
      notify.success('Đã kích hoạt provider')
    } catch (reason) { notify.error('Không thể kích hoạt provider', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function loadModels(id: string) {
    setLoadingModels(id)
    try {
      const result = await api<{ models: string[] }>(`/api/providers/${id}/models`)
      setModels(previous => ({ ...previous, [id]: result.models }))
      notify.success(`Đã tải ${result.models.length} model`)
    } catch (reason) { notify.error('Không tải được model', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
    finally { setLoadingModels(null) }
  }

  async function setModel(id: string, model: string) {
    try {
      await api(`/api/providers/${id}/model`, { method: 'POST', body: JSON.stringify({ model }) })
      await load()
      notify.success(`Đã chọn model ${model}`)
    } catch (reason) { notify.error('Không thể đổi model', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function saveThreshold() {
    try {
      await api('/api/settings/threshold', { method: 'POST', body: JSON.stringify({ threshold }) })
      notify.success('Đã lưu ngưỡng bằng chứng')
    } catch (reason) { notify.error('Không thể lưu ngưỡng', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function createUser(event: FormEvent) {
    event.preventDefault()
    try {
      await api('/api/users', { method: 'POST', body: JSON.stringify({ username, password, role }) })
      setUsername(''); setPassword(''); await load()
      notify.success('Đã tạo người dùng')
    } catch (reason) { notify.error('Không thể tạo người dùng', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function changeRole(user: UserRow, nextRole: Role) {
    try {
      await api(`/api/users/${user.id}`, { method: 'PATCH', body: JSON.stringify({ role: nextRole }) })
      await load()
      notify.success(`Đã đổi vai trò của ${user.username}`)
    } catch (reason) { notify.error('Không thể đổi vai trò', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function resetPassword(user: UserRow) {
    const next = window.prompt(`Mật khẩu mới cho ${user.username} (ít nhất 10 ký tự)`)
    if (!next) return
    try {
      await api(`/api/users/${user.id}`, { method: 'PATCH', body: JSON.stringify({ password: next }) })
      await load()
      notify.success(`Đã đổi mật khẩu cho ${user.username}`)
    } catch (reason) { notify.error('Không thể đổi mật khẩu', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  async function removeUser(user: UserRow) {
    try {
      await api(`/api/users/${user.id}`, { method: 'DELETE' })
      await load()
      notify.success(`Đã xóa ${user.username}`)
    } catch (reason) { notify.error('Không thể xóa người dùng', { description: reason instanceof Error ? reason.message : 'Yêu cầu thất bại' }) }
  }

  return (
    <div className="page admin-page" aria-busy={loading}>
      <div className="page-heading"><div><h1>Quản trị</h1><p>Provider, ngưỡng bằng chứng, người dùng và audit không chứa nội dung nhạy cảm.</p></div></div>
      <section className="panel admin-section">
        <div className="panel-title"><b>LLM runtime</b><span>Key chỉ đọc từ biến môi trường</span></div>
        <div className="provider-grid" aria-busy={loading}>{loading && !providers.length ? <div className="empty-state col-span-full" role="status">Đang tải cấu hình runtime…</div> : !providers.length ? <div className="empty-state col-span-full">Chưa có provider.</div> : providers.map(provider => (
          <article className={`provider-card ${provider.active ? 'active' : ''}`} key={provider.id}>
            <div><b>{provider.label}</b><span className="mono">{provider.base_url}</span></div>
            <span className="key-env mono">{provider.key_env || 'NO KEY'}</span>
            <div className="provider-actions">
              <button className="button secondary" disabled={loadingModels === provider.id} onClick={() => loadModels(provider.id)}>{loadingModels === provider.id ? 'Đang tải…' : 'Tải model'}</button>
              {!provider.active && <button className="button" disabled={loading} onClick={() => activate(provider.id)}>Kích hoạt</button>}
            </div>
            <select aria-label={`Model ${provider.label}`} disabled={loadingModels === provider.id} value={provider.model} onChange={event => setModel(provider.id, event.target.value)}>
              <option value={provider.model}>{provider.model}</option>
              {(models[provider.id] || []).filter(model => model !== provider.model).map(model => <option key={model}>{model}</option>)}
            </select>
          </article>
        ))}</div>
      </section>
      <section className="panel admin-section threshold-row">
        <div><b>Ngưỡng bằng chứng</b><p>Mặc định 0.78 · mức thận trọng tối đa khuyến nghị 0.84.</p></div>
        <input aria-label="Ngưỡng bằng chứng" type="range" min="0" max="1" step="0.01" disabled={loading} value={threshold} onChange={event => setThreshold(Number(event.target.value))} />
        <span className="threshold-value mono">{threshold.toFixed(2)}</span>
        <button className="button" disabled={loading} onClick={saveThreshold}>Lưu</button>
      </section>
      <section className="panel admin-section">
        <div className="panel-title"><b>Người dùng</b><span>{users.length} tài khoản</span></div>
        <form className="user-create" onSubmit={createUser}>
          <input aria-label="Tên đăng nhập mới" placeholder="Tên đăng nhập" disabled={loading} value={username} onChange={event => setUsername(event.target.value)} required />
          <input aria-label="Mật khẩu người dùng mới" type="password" placeholder="Mật khẩu ≥ 10 ký tự" disabled={loading} value={password} onChange={event => setPassword(event.target.value)} required minLength={10} />
          <select aria-label="Vai trò người dùng mới" disabled={loading} value={role} onChange={event => setRole(event.target.value as Role)}><option value="viewer">viewer</option><option value="user">user</option><option value="admin">admin</option></select>
          <button className="button" disabled={loading}>Tạo</button>
        </form>
        <div className="table-wrap"><table><thead><tr><th>Tài khoản</th><th>Vai trò</th><th>Ngày tạo</th><th><span className="sr-only">Thao tác</span></th></tr></thead><tbody>
          {!users.length ? <tr><td className="empty-state" colSpan={4}>{loading ? 'Đang tải người dùng…' : 'Chưa có tài khoản.'}</td></tr> : users.map(user => <tr key={user.id}><td><b>{user.username}</b></td><td><select aria-label={`Vai trò ${user.username}`} disabled={loading} value={user.role} onChange={event => changeRole(user, event.target.value as Role)}><option>viewer</option><option>user</option><option>admin</option></select></td><td className="mono">{new Date(user.created_at).toLocaleString('vi-VN')}</td><td className="actions"><button className="button secondary" disabled={loading} onClick={() => resetPassword(user)}>Đổi mật khẩu</button><button className="button danger" disabled={loading} onClick={() => removeUser(user)}>Xóa</button></td></tr>)}
        </tbody></table></div>
      </section>
      <section className="panel admin-section">
        <div className="panel-title"><b>Audit</b><span>Không lưu câu hỏi, câu trả lời, tài liệu hoặc token</span></div>
        <div className="table-wrap"><table><thead><tr><th>Thời gian</th><th>Người dùng</th><th>Action</th><th>Resource</th></tr></thead><tbody>
          {!audit.length ? <tr><td className="empty-state" colSpan={4}>{loading ? 'Đang tải audit…' : 'Chưa có bản ghi audit.'}</td></tr> : audit.map(row => <tr key={row.id}><td className="mono">{new Date(row.timestamp).toLocaleString('vi-VN')}</td><td>{row.username}</td><td className="mono">{row.action}</td><td className="mono">{row.resource_id || '—'}</td></tr>)}
        </tbody></table></div>
      </section>
    </div>
  )
}
