import { useEffect, useRef, useState } from 'react'

interface Props {
  onMissionStart?: (missionId: string) => void
  onMissionComplete?: () => void
}

interface StatusResponse {
  running: boolean
  mission_id: string | null
  log_tail: string[]
}

interface StartResponse {
  status: string
  sim: boolean
  mission_id: string
}

function logColor(line: string): string {
  if (line.includes('ERROR') || line.includes('Fehler')) return '#f44336'
  if (line.includes('WARNING')) return '#ff9800'
  if (line === '__END__') return '#4caf50'
  return '#aaa'
}

export default function ChatPanel({ onMissionStart, onMissionComplete }: Props) {
  type MissionMode = 'sim' | 'camtest' | 'real'
  const [prompt, setPrompt] = useState('')
  const [circleRadius, setCircleRadius] = useState<number | ''>(5)
  const [missionMode, setMissionMode] = useState<MissionMode>('camtest')
  const [isRunning, setIsRunning] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  // ── Check running state on mount ─────────────────────────────────────────
  useEffect(() => {
    fetch('/api/mission/status')
      .then((r) => r.json() as Promise<StatusResponse>)
      .then((data) => {
        setIsRunning(data.running)
        if (data.log_tail.length > 0) setLogLines(data.log_tail)
        if (data.running) {
          // Reloaded mid-run: adopt the mission already in flight so the board
          // shows *its* findings rather than the previously selected mission's.
          if (data.mission_id) onMissionStart?.(data.mission_id)
          connectSSE()
        }
      })
      .catch(console.error)

    return () => {
      esRef.current?.close()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Auto-scroll log ───────────────────────────────────────────────────────
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logLines])

  // ── SSE connection ────────────────────────────────────────────────────────
  function connectSSE() {
    esRef.current?.close()
    const es = new EventSource('/api/mission/log')

    es.onmessage = (ev) => {
      const line = ev.data as string
      if (line === '__END__') {
        es.close()
        esRef.current = null
        setIsRunning(false)
        onMissionComplete?.()
        return
      }
      setLogLines((prev) => [...prev.slice(-199), line])
    }

    es.onerror = () => {
      es.close()
      esRef.current = null
      setIsRunning(false)
    }

    esRef.current = es
  }

  // ── Handlers ──────────────────────────────────────────────────────────────
  async function handleStart() {
    if (isRunning || !prompt.trim()) return
    setError(null)
    setLogLines([])

    const body: Record<string, unknown> = {
      prompt: prompt.trim(),
      sim_mode: missionMode === 'sim',
      detection_test: missionMode === 'camtest',
    }
    if (missionMode !== 'camtest' && circleRadius !== '' && circleRadius > 0)
      body.circle_radius_m = circleRadius

    const res = await fetch('/api/mission/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (res.ok) {
      const data = (await res.json()) as StartResponse
      setIsRunning(true)
      onMissionStart?.(data.mission_id)  // open the new, empty board
      connectSSE()
    } else {
      const data = await res.json().catch(() => ({}))
      setError((data as { detail?: string }).detail ?? `HTTP ${res.status}`)
    }
  }

  async function handleStop() {
    await fetch('/api/mission/stop', { method: 'POST' }).catch(console.error)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      padding: '10px', gap: '8px', boxSizing: 'border-box',
    }}>
      {/* Prompt input */}
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Suche 10m um mich nach Müll…"
        disabled={isRunning}
        rows={3}
        style={{
          width: '100%', background: '#111', color: '#ddd', border: '1px solid #333',
          borderRadius: 4, padding: '6px 8px', fontSize: 12, resize: 'vertical',
          fontFamily: 'inherit', flexShrink: 0,
        }}
        onKeyDown={(e) => { if (e.key === 'Enter' && e.ctrlKey) void handleStart() }}
      />

      {/* Mode selector */}
      <div style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
        {([
          ['camtest', 'Kamera-Test', '#1565c0', 'Detektor läuft, kein Roboter nötig'],
          ['sim',     'Karten-Sim',  '#2e7d32', 'Fake-Navigation auf Karte, kein Detektor'],
          ['real',    'Roboter',     '#6a1b9a', 'Vollbetrieb mit echtem Robot'],
        ] as [string, string, string, string][]).map(([val, label, color, title]) => (
          <button
            key={val}
            title={title}
            onClick={() => !isRunning && setMissionMode(val as MissionMode)}
            style={{
              flex: 1, padding: '4px 0', fontSize: 10, borderRadius: 3, border: 'none',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              background: missionMode === val ? color : '#252525',
              color: missionMode === val ? '#fff' : '#555',
              fontWeight: missionMode === val ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Controls row */}
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
        <label style={{ fontSize: 11, color: '#666', whiteSpace: 'nowrap' }}>Radius (m)</label>
        <input
          type="number"
          value={circleRadius}
          onChange={(e) => setCircleRadius(e.target.value === '' ? '' : Number(e.target.value))}
          disabled={isRunning}
          min={1}
          max={50}
          style={{
            width: 56, background: '#111', color: '#ddd', border: '1px solid #333',
            borderRadius: 4, padding: '4px 6px', fontSize: 12,
          }}
        />

        <button
          onClick={() => void handleStart()}
          disabled={isRunning || !prompt.trim()}
          style={{
            flex: 1, padding: '5px 0', fontSize: 12, borderRadius: 4,
            border: 'none', cursor: isRunning || !prompt.trim() ? 'not-allowed' : 'pointer',
            background: isRunning || !prompt.trim() ? '#2a2a2a' : '#1b5e20',
            color: isRunning || !prompt.trim() ? '#444' : '#81c784',
          }}
        >
          Starten
        </button>

        {isRunning && (
          <button
            onClick={() => void handleStop()}
            style={{
              padding: '5px 10px', fontSize: 12, borderRadius: 4,
              border: 'none', cursor: 'pointer', background: '#3a1a1a', color: '#f44336',
            }}
          >
            Stopp
          </button>
        )}
      </div>

      {/* Status indicator */}
      {isRunning && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, flexShrink: 0 }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', background: missionMode === 'sim' ? '#4caf50' : missionMode === 'camtest' ? '#2196f3' : '#ff9800',
            display: 'inline-block', animation: 'pulse 1.2s infinite',
          }} />
          <span style={{ color: missionMode === 'sim' ? '#4caf50' : missionMode === 'camtest' ? '#2196f3' : '#ff9800', marginLeft: 4 }}>
            {missionMode === 'sim' ? 'Karten-Sim läuft…' : missionMode === 'camtest' ? 'Kamera-Test läuft…' : 'Mission läuft…'}
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ fontSize: 11, color: '#f44336', flexShrink: 0 }}>
          Fehler: {error}
        </div>
      )}

      {/* Log area */}
      <div style={{
        flex: 1, minHeight: 0, overflow: 'auto',
        background: '#0a0a0a', border: '1px solid #222', borderRadius: 4,
        padding: '6px 8px', fontFamily: 'monospace', fontSize: 10, lineHeight: 1.5,
      }}>
        {logLines.length === 0 && !isRunning && (
          <span style={{ color: '#333' }}>Kein Log</span>
        )}
        {logLines.map((line, i) => (
          <div key={i} style={{ color: logColor(line), wordBreak: 'break-all' }}>
            {line}
          </div>
        ))}
        <div ref={logEndRef} />
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  )
}
