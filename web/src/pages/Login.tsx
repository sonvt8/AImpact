import { FormEvent, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { notify } from '../notify'

export default function Login() {
  const { user, signIn } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/" replace />

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    try {
      await signIn(username, password)
      navigate('/')
    } catch (reason) {
      notify.error('Đăng nhập thất bại', { description: reason instanceof Error ? reason.message : 'Không thể xác thực tài khoản' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-brand"><span>AI</span><div><b>AImpact</b><small>Evidence-first operations search</small></div></div>
        <div className="login-copy">
          <span className="eyebrow">TRUY CẬP NỘI BỘ</span>
          <h1>Đăng nhập hệ thống</h1>
          <p>Câu trả lời chỉ được tạo khi có đủ bằng chứng và luôn kèm vị trí nguồn.</p>
        </div>
        <form onSubmit={submit} className="login-form" aria-busy={busy}>
          <div className="field"><label htmlFor="username">Tên đăng nhập</label><input id="username" autoFocus autoComplete="username" disabled={busy} value={username} onChange={event => setUsername(event.target.value)} /></div>
          <div className="field"><label htmlFor="password">Mật khẩu</label><input id="password" type="password" autoComplete="current-password" disabled={busy} value={password} onChange={event => setPassword(event.target.value)} /></div>
          <button className="button" disabled={busy}>{busy ? 'Đang xác thực…' : 'Đăng nhập'}</button>
        </form>
      </div>
      <div className="login-grid" aria-hidden="true" />
    </div>
  )
}
