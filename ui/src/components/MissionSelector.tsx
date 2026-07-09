import type { MissionRow } from '../types'

interface Props {
  missions: MissionRow[]
  selectedId: string | null
  onSelect: (id: string) => void
}

function fmtDate(ns: number | null): string {
  if (!ns) return '—'
  return new Date(ns / 1e6).toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' })
}

export default function MissionSelector({ missions, selectedId, onSelect }: Props) {
  if (missions.length === 0) {
    return <span style={{ color: '#555', fontSize: 12 }}>Keine Missionen</span>
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Mission
      </span>
      <select
        value={selectedId ?? ''}
        onChange={(e) => onSelect(e.target.value)}
        style={{
          background: '#252525', color: '#ccc', border: '1px solid #333',
          borderRadius: 4, padding: '4px 8px', fontSize: 12,
        }}
      >
        {missions.map((m) => {
          const validated = m.status_counts['validated'] ?? 0
          const total = Object.values(m.status_counts).reduce((a, b) => a + b, 0)
          const cov = m.coverage_fraction != null ? `${(m.coverage_fraction * 100).toFixed(0)}%` : '?'
          return (
            <option key={m.mission_id} value={m.mission_id}>
              {fmtDate(m.started_ns)} — {validated}/{total} Findings — {cov} Coverage
            </option>
          )
        })}
      </select>
    </div>
  )
}
