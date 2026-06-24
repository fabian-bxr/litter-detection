import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import type { FindingRow, MapConfig, StateMessage } from '../types'

interface Props {
  findings: FindingRow[]
  onFindingSelect: (trackId: number) => void
  highlightedTrackId: number | null
}

export default function MapPanel({ findings, onFindingSelect, highlightedTrackId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const robotRef = useRef<L.CircleMarker | null>(null)
  const pathRef = useRef<L.Polyline | null>(null)
  const plannedRef = useRef<L.Polyline | null>(null)
  const litterRefs = useRef<L.CircleMarker[]>([])
  const [mapReady, setMapReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── Initialise Leaflet map once ──────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      crs: L.CRS.Simple,
      attributionControl: false,
      zoomControl: true,
    })
    mapRef.current = map

    fetch('/api/map/config')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<MapConfig>
      })
      .then((cfg) => {
        const { origin_x, origin_y, resolution, width_px, height_px } = cfg
        if (!width_px || !height_px) throw new Error('Kartendimensionen unbekannt')

        const w = width_px * resolution   // metres
        const h = height_px * resolution  // metres

        // Leaflet CRS.Simple: [lat, lng] = [y, x]
        const bottomLeft: L.LatLngTuple = [origin_y, origin_x]
        const topRight: L.LatLngTuple   = [origin_y + h, origin_x + w]
        const bounds: L.LatLngBoundsExpression = [bottomLeft, topRight]

        L.imageOverlay('/api/map/image', bounds).addTo(map)
        map.fitBounds(bounds)

        pathRef.current = L.polyline([], {
          color: '#4caf50', opacity: 0.75, weight: 2,
        }).addTo(map)

        plannedRef.current = L.polyline([], {
          color: '#ff9800', opacity: 0.9, weight: 2, dashArray: '6 4',
        }).addTo(map)

        // Robot starts at origin; will be moved by /ws/state
        robotRef.current = L.circleMarker(bottomLeft, {
          radius: 7, color: '#2196f3', fillColor: '#2196f3', fillOpacity: 1, weight: 2,
        }).addTo(map)

        setMapReady(true)
      })
      .catch((err: unknown) => setError(String(err)))

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // ── State WebSocket ──────────────────────────────────────────────────────
  useEffect(() => {
    let ws: WebSocket | null = null
    let cancelled = false

    function connect() {
      if (cancelled) return
      ws = new WebSocket(`ws://${location.host}/ws/state`)

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data as string) as StateMessage
        if (msg.error || !mapRef.current) return

        if (msg.pose && robotRef.current) {
          robotRef.current.setLatLng([msg.pose.y, msg.pose.x])
        }
        if (pathRef.current) {
          // state sends [x,y]; Leaflet wants [y,x]
          pathRef.current.setLatLngs(
            msg.path_history.map(([x, y]) => [y, x] as L.LatLngTuple),
          )
        }
        if (plannedRef.current) {
          plannedRef.current.setLatLngs(
            msg.planned_path.map(([x, y]) => [y, x] as L.LatLngTuple),
          )
        }
      }

      ws.onclose = () => {
        if (!cancelled) setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      cancelled = true
      ws?.close()
    }
  }, [])

  // ── Litter markers ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapReady || !mapRef.current) return

    litterRefs.current.forEach((m) => m.remove())
    litterRefs.current = []

    findings.forEach((f) => {
      if (f.robot_x === null || f.robot_y === null) return
      const highlighted = f.track_id === highlightedTrackId
      const marker = L.circleMarker([f.robot_y, f.robot_x], {
        radius: highlighted ? 9 : 6,
        color: highlighted ? '#ffeb3b' : '#f44336',
        fillColor: highlighted ? '#ffeb3b' : '#f44336',
        fillOpacity: 0.85,
        weight: 2,
      })
      marker.on('click', () => onFindingSelect(f.track_id))
      marker.bindPopup(
        `<strong>${f.category ?? 'Unbekannt'}</strong><br/>` +
        `Konfidenz: ${f.confidence != null ? (f.confidence * 100).toFixed(0) : '?'}%<br/>` +
        (f.description ?? ''),
      )
      marker.addTo(mapRef.current!)
      litterRefs.current.push(marker)
    })
  }, [findings, highlightedTrackId, onFindingSelect, mapReady])

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />

      {/* Legend */}
      {mapReady && (
        <div style={{
          position: 'absolute', bottom: 8, left: 8, zIndex: 1000,
          background: 'rgba(0,0,0,0.65)', borderRadius: 4, padding: '5px 10px',
          fontSize: 11, display: 'flex', flexDirection: 'column', gap: 3,
        }}>
          <span><span style={{ color: '#4caf50' }}>—</span> Gefahren</span>
          <span><span style={{ color: '#ff9800' }}>- -</span> Geplant</span>
          <span><span style={{ color: '#f44336' }}>●</span> Müll</span>
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
          justifyContent: 'center', background: '#1a1a1a',
          color: '#f44336', fontSize: 13, padding: 20, textAlign: 'center',
        }}>
          Karte nicht verfügbar: {error}
        </div>
      )}
    </div>
  )
}
