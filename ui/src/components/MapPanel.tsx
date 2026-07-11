import { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import type { CoverageStats, FindingRow, MapConfig, StateMessage } from '../types'

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
  const coverageRef = useRef<L.ImageOverlay | null>(null)
  const boundsRef = useRef<L.LatLngBoundsExpression | null>(null)
  const lastOverlaySeqRef = useRef<number>(-1)
  const showOverlayRef = useRef<boolean>(true)
  const litterRefs = useRef<L.CircleMarker[]>([])
  const [mapReady, setMapReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [coverage, setCoverage] = useState<CoverageStats | null>(null)

  // ── Initialise Leaflet map once ──────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      crs: L.CRS.Simple,
      attributionControl: false,
      zoomControl: true,
    })
    mapRef.current = map

    // Dedicated stacking order: static map < coverage overlay < robot/path/litter.
    map.createPane('basemap')
    map.getPane('basemap')!.style.zIndex = '250'
    map.createPane('coverage')
    map.getPane('coverage')!.style.zIndex = '350'

    fetch('/api/map/config')
      .then(async (r) => {
        if (!r.ok) {
          // The backend explains itself in `detail` (missing map file, MOLA
          // unreachable, …) — a bare "HTTP 404" tells nobody anything.
          const detail = await r.json().then(
            (b: { detail?: string }) => b.detail,
            () => undefined,
          )
          throw new Error(detail ? `HTTP ${r.status}: ${detail}` : `HTTP ${r.status}`)
        }
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
        boundsRef.current = bounds

        L.imageOverlay('/api/map/image', bounds, { pane: 'basemap' }).addTo(map)
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
      coverageRef.current = null
      boundsRef.current = null
    }
  }, [])

  // ── State WebSocket ──────────────────────────────────────────────────────
  useEffect(() => {
    let ws: WebSocket | null = null
    let cancelled = false

    // Add / update / drop the coverage overlay image when the sequence bumps.
    function syncOverlay(seq: number, hasCoverage: boolean) {
      const map = mapRef.current
      const bounds = boundsRef.current
      if (!map || !bounds) return

      if (!hasCoverage) {
        coverageRef.current?.remove()
        coverageRef.current = null
        return
      }
      const url = `/api/map/coverage.png?seq=${seq}`
      if (coverageRef.current) {
        coverageRef.current.setUrl(url)
      } else {
        coverageRef.current = L.imageOverlay(url, bounds, {
          pane: 'coverage',
          opacity: showOverlayRef.current ? 1 : 0,
          interactive: false,
        }).addTo(map)
      }
    }

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

        // Coverage overlay: refetch the PNG only when its version changes.
        if (typeof msg.overlay_seq === 'number' && msg.overlay_seq !== lastOverlaySeqRef.current) {
          lastOverlaySeqRef.current = msg.overlay_seq
          syncOverlay(msg.overlay_seq, !!msg.coverage)
        }
        setCoverage(msg.coverage ?? null)
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

  // ── Overlay visibility toggle ─────────────────────────────────────────────
  useEffect(() => {
    showOverlayRef.current = showOverlay
    coverageRef.current?.setOpacity(showOverlay ? 1 : 0)
  }, [showOverlay])

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

  const coveragePct = coverage ? Math.round(coverage.fraction * 100) : null

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />

      {/* Coverage badge */}
      {mapReady && coverage && (
        <div style={{
          position: 'absolute', top: 8, right: 8, zIndex: 1000,
          background: 'rgba(0,0,0,0.68)', borderRadius: 6, padding: '7px 11px',
          minWidth: 92, backdropFilter: 'blur(2px)',
          boxShadow: '0 1px 6px rgba(0,0,0,0.4)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
            <span style={{ fontSize: 22, fontWeight: 700, color: '#66bb6a', lineHeight: 1 }}>
              {coveragePct}%
            </span>
            <span style={{ fontSize: 10, color: '#999' }}>abgedeckt</span>
          </div>
          {/* progress bar */}
          <div style={{
            marginTop: 5, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.12)',
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%', width: `${coveragePct ?? 0}%`,
              background: 'linear-gradient(90deg,#43a047,#66bb6a)',
              transition: 'width 0.4s ease',
            }} />
          </div>
          <div style={{ fontSize: 9.5, color: '#777', marginTop: 4 }}>
            Suchgebiet {coverage.reachable_m2.toFixed(1)} m²
          </div>
        </div>
      )}

      {/* Legend */}
      {mapReady && (
        <div style={{
          position: 'absolute', bottom: 8, left: 8, zIndex: 1000,
          background: 'rgba(0,0,0,0.65)', borderRadius: 4, padding: '5px 10px',
          fontSize: 11, display: 'flex', flexDirection: 'column', gap: 3,
        }}>
          <span><span style={{ color: '#3fd88a' }}>■</span> Gesehen</span>
          <span><span style={{ color: '#3884ff' }}>■</span> Offen</span>
          <span><span style={{ color: '#22d3ee' }}>◯</span> Suchgebiet</span>
          <span><span style={{ color: '#4caf50' }}>—</span> Gefahren</span>
          <span><span style={{ color: '#ff9800' }}>- -</span> Geplant</span>
          <span><span style={{ color: '#f44336' }}>●</span> Müll</span>
          <label style={{
            display: 'flex', alignItems: 'center', gap: 5, marginTop: 3,
            paddingTop: 4, borderTop: '1px solid rgba(255,255,255,0.12)',
            cursor: 'pointer', color: '#bbb',
          }}>
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
              style={{ margin: 0, cursor: 'pointer' }}
            />
            Abdeckung
          </label>
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
