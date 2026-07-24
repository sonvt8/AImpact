import { lazy, Suspense } from 'react'
import { Navigate, NavLink, Outlet, Route, Routes } from 'react-router-dom'
import { useAuth } from './lib/auth'
import type { Role } from './lib/types'
import LoadingOverlay from './components/LoadingOverlay'
import ThemeToggle from './components/ThemeToggle'

const Login = lazy(() => import('./pages/Login'))
const Chat = lazy(() => import('./pages/Chat'))
const Documents = lazy(() => import('./pages/Documents'))
const Stats = lazy(() => import('./pages/Stats'))
const Admin = lazy(() => import('./pages/Admin'))

function Protected({ role }: { role?: Role }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingOverlay label="Đang khôi phục phiên" />
  if (!user) return <Navigate to="/login" replace />
  if (role && user.role !== role) return <Navigate to="/" replace />
  return <Outlet />
}

function Layout() {
  const { user, signOut } = useAuth()
  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to="/" className="brand" aria-label="AImpact chat">
          <span className="brand-mark">AI</span>
          <span><strong>AImpact</strong><small>RAG vận hành kỹ thuật</small></span>
        </NavLink>
        <nav aria-label="Điều hướng chính">
          <NavLink to="/">Tra cứu</NavLink>
          <NavLink to="/stats">Thống kê</NavLink>
          {user?.role === 'admin' && <NavLink to="/documents">Tài liệu</NavLink>}
          {user?.role === 'admin' && <NavLink to="/admin">Quản trị</NavLink>}
        </nav>
        <ThemeToggle />
        <div className="user-menu">
          <span><b>{user?.username}</b><small>{user?.role}</small></span>
          <button className="button ghost" onClick={() => signOut()}>Đăng xuất</button>
        </div>
      </header>
      <main><Outlet /></main>
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<LoadingOverlay label="Đang tải trang" />}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<Protected />}>
          <Route element={<Layout />}>
            <Route index element={<Chat />} />
            <Route path="stats" element={<Stats />} />
            <Route element={<Protected role="admin" />}>
              <Route path="documents" element={<Documents />} />
              <Route path="admin" element={<Admin />} />
            </Route>
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}
