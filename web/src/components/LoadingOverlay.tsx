export default function LoadingOverlay({ label }: { label: string }) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-card">
        <span className="loading-spinner" aria-hidden="true" />
        <span>{label}</span>
      </div>
    </div>
  )
}
