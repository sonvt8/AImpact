type NotifyOptions = { description?: string }

type ToastKind = 'error' | 'success'

const AUTO_DISMISS_MS = 4500
let region: HTMLDivElement | null = null

function getRegion() {
  if (region?.isConnected) return region
  region = document.createElement('div')
  region.className = 'toast-region'
  region.setAttribute('role', 'status')
  region.setAttribute('aria-live', 'polite')
  region.setAttribute('aria-label', 'Thông báo')
  document.body.append(region)
  return region
}

function show(kind: ToastKind, message: string, options?: NotifyOptions) {
  if (typeof document === 'undefined') return

  const toast = document.createElement('div')
  toast.className = `toast toast-${kind}`
  toast.setAttribute('role', kind === 'error' ? 'alert' : 'status')
  toast.setAttribute('aria-live', kind === 'error' ? 'assertive' : 'polite')
  toast.setAttribute('aria-atomic', 'true')

  const accent = document.createElement('span')
  accent.className = 'toast-accent'
  accent.setAttribute('aria-hidden', 'true')

  const content = document.createElement('div')
  content.className = 'toast-content'

  const title = document.createElement('strong')
  title.className = 'toast-title'
  title.textContent = message
  content.append(title)

  if (options?.description) {
    const description = document.createElement('p')
    description.className = 'toast-description'
    description.textContent = options.description
    content.append(description)
  }

  const close = document.createElement('button')
  close.className = 'toast-close'
  close.type = 'button'
  close.setAttribute('aria-label', 'Đóng thông báo')
  close.textContent = '×'

  toast.append(accent, content, close)
  getRegion().append(toast)

  const dismiss = () => {
    toast.remove()
    if (region && !region.childElementCount) {
      region.remove()
      region = null
    }
  }
  const timeout = window.setTimeout(dismiss, AUTO_DISMISS_MS)
  close.addEventListener('click', () => {
    window.clearTimeout(timeout)
    dismiss()
  })
}

function error(message: string, options?: NotifyOptions) {
  show('error', message, options)
}

function success(message: string, options?: NotifyOptions) {
  show('success', message, options)
}

export const notify = { error, success }
