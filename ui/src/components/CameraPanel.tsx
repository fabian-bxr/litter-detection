import { useEffect, useRef, useState } from 'react'

type Status = 'connecting' | 'connected' | 'no_signal' | 'zenoh_unavailable'

const STATUS_LABEL: Record<Status, string> = {
  connecting: 'Verbinde…',
  connected: 'Live',
  no_signal: 'Kein Signal',
  zenoh_unavailable: 'Zenoh nicht verfügbar',
}

const STATUS_COLOR: Record<Status, string> = {
  connecting: '#888',
  connected: '#4caf50',
  no_signal: '#f44336',
  zenoh_unavailable: '#ff9800',
}

export default function CameraPanel() {
  const imgRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const prevUrlRef = useRef<string | null>(null)
  const [status, setStatus] = useState<Status>('connecting')

  useEffect(() => {
    let cancelled = false
    // Track status in a closure-local var to avoid stale captures in callbacks.
    let currentStatus: Status = 'connecting'

    function setS(s: Status) {
      currentStatus = s
      setStatus(s)
    }

    function connect() {
      if (cancelled) return
      setS('connecting')

      const ws = new WebSocket(`ws://${location.host}/ws/camera`)
      wsRef.current = ws

      ws.onmessage = (ev) => {
        if (ev.data instanceof Blob) {
          if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)
          const url = URL.createObjectURL(ev.data)
          prevUrlRef.current = url
          if (imgRef.current) imgRef.current.src = url
          setS('connected')
        } else {
          try {
            const msg = JSON.parse(ev.data as string) as Record<string, unknown>
            if (msg.error === 'zenoh_unavailable') setS('zenoh_unavailable')
            // keepalive messages: ignore
          } catch { /* ignore */ }
        }
      }

      ws.onerror = () => setS('no_signal')

      ws.onclose = () => {
        if (!cancelled && currentStatus !== 'zenoh_unavailable') {
          setS('no_signal')
          setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      cancelled = true
      wsRef.current?.close()
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)
    }
  }, [])

  return (
    <div style={{ position: 'relative', height: '100%', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <img
        ref={imgRef}
        alt="Kamera-Feed"
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
      />
      <div style={{
        position: 'absolute', top: 8, left: 8,
        background: 'rgba(0,0,0,0.65)', borderRadius: 4, padding: '3px 9px',
        fontSize: 11, display: 'flex', alignItems: 'center', gap: 6,
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: STATUS_COLOR[status], display: 'inline-block', flexShrink: 0,
        }} />
        {STATUS_LABEL[status]}
      </div>
    </div>
  )
}
