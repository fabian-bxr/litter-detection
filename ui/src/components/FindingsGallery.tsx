import { useEffect, useRef, useState } from 'react'
import type { FindingRow } from '../types'

const LITTER_CATEGORIES = [
  'plastic', 'paper', 'cardboard', 'metal', 'glass',
  'organic', 'cigarette', 'textile', 'other',
] as const

const CATEGORY_COLOR: Record<string, string> = {
  plastic: '#2196f3', paper: '#9e9e9e', cardboard: '#795548',
  metal: '#607d8b', glass: '#00bcd4', organic: '#4caf50',
  cigarette: '#ff5722', textile: '#9c27b0', other: '#ff9800',
}

const STATUS_COLOR: Record<string, string> = {
  validated: '#4caf50', rejected: '#f44336', error: '#ff9800',
}

type SortKey = 'confidence' | 'validated_at' | 'area_px'

interface EditState { category: string; status: string }

interface Props {
  missionId: string | null
  findings: FindingRow[]
  onFindingSelect: (trackId: number) => void
  highlightedTrackId: number | null
  onDelete: (missionId: string, trackId: number) => void
  onUpdate: (updated: FindingRow) => void
}

function cropUrl(f: FindingRow) {
  return `/api/findings/${f.mission_id}/${f.track_id}/image?type=crop`
}
function contextUrl(f: FindingRow) {
  return `/api/findings/${f.mission_id}/${f.track_id}/image?type=context`
}
function fmtNs(ns: number) {
  return new Date(ns / 1e6).toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'medium' })
}

export default function FindingsGallery({
  missionId,
  findings,
  onFindingSelect,
  highlightedTrackId,
  onDelete,
  onUpdate,
}: Props) {
  const [filterStatus, setFilterStatus] = useState<'all' | 'validated' | 'rejected' | 'error'>('all')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [sortBy, setSortBy] = useState<SortKey>('confidence')
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null) // `${missionId}/${trackId}`
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [edit, setEdit] = useState<EditState>({ category: '', status: '' })
  const [modalFinding, setModalFinding] = useState<FindingRow | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // ── Filter + sort ─────────────────────────────────────────────────────────
  const visible = findings
    .filter((f) => filterStatus === 'all' || f.status === filterStatus)
    .filter((f) => filterCategory === 'all' || f.category === filterCategory)
    .sort((a, b) => {
      const va =
        sortBy === 'confidence' ? (a.confidence ?? 0)
        : sortBy === 'validated_at' ? a.validated_at_ns
        : a.area_px
      const vb =
        sortBy === 'confidence' ? (b.confidence ?? 0)
        : sortBy === 'validated_at' ? b.validated_at_ns
        : b.area_px
      return sortDir === 'desc' ? vb - va : va - vb
    })

  // ── Scroll highlighted card into view ─────────────────────────────────────
  useEffect(() => {
    if (highlightedTrackId === null) return
    const el = document.getElementById(`finding-${highlightedTrackId}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
  }, [highlightedTrackId])

  // ── Delete ────────────────────────────────────────────────────────────────
  async function handleDelete(mId: string, tId: number) {
    const res = await fetch(`/api/findings/${mId}/${tId}`, { method: 'DELETE' })
    if (res.ok || res.status === 404) {
      onDelete(mId, tId)
    }
    setDeleteConfirm(null)
  }

  // ── Edit save ────────────────────────────────────────────────────────────
  async function handleSave(mId: string, tId: number) {
    const body: Record<string, string> = {}
    if (edit.category) body.category = edit.category
    if (edit.status) body.status = edit.status
    if (Object.keys(body).length === 0) { setEditingKey(null); return }
    const res = await fetch(`/api/findings/${mId}/${tId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (res.ok) {
      const updated = await res.json() as FindingRow
      onUpdate(updated)
    }
    setEditingKey(null)
  }

  function startEdit(f: FindingRow, e: React.MouseEvent) {
    e.stopPropagation()
    setDeleteConfirm(null)
    setEditingKey(`${f.mission_id}/${f.track_id}`)
    setEdit({ category: f.category ?? '', status: f.status })
  }

  if (!missionId) {
    return (
      <div className="gallery-section">
        <div className="gallery-empty">Keine Mission ausgewählt</div>
      </div>
    )
  }

  return (
    <>
      <div className="gallery-section">
        {/* Toolbar */}
        <div className="gallery-toolbar">
          <span style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: '0.08em', marginRight: 4 }}>
            {visible.length} / {findings.length} Findings
          </span>

          <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value as typeof filterStatus)}>
            <option value="all">Alle Status</option>
            <option value="validated">Validiert</option>
            <option value="rejected">Abgelehnt</option>
            <option value="error">Fehler</option>
          </select>

          <select value={filterCategory} onChange={(e) => setFilterCategory(e.target.value)}>
            <option value="all">Alle Kategorien</option>
            {LITTER_CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortKey)}>
            <option value="confidence">Konfidenz</option>
            <option value="validated_at">Zeit</option>
            <option value="area_px">Fläche</option>
          </select>

          <button onClick={() => setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))}>
            {sortDir === 'desc' ? '↓' : '↑'}
          </button>
        </div>

        {/* Card grid */}
        {visible.length === 0 ? (
          <div className="gallery-empty">
            {findings.length === 0 ? 'Keine Findings in dieser Mission' : 'Keine Findings mit diesen Filtern'}
          </div>
        ) : (
          <div className="gallery-grid" ref={scrollRef}>
            {visible.map((f) => {
              const key = `${f.mission_id}/${f.track_id}`
              const isHighlighted = f.track_id === highlightedTrackId
              const isDeleting = deleteConfirm === key
              const isEditing = editingKey === key

              return (
                <div
                  id={`finding-${f.track_id}`}
                  key={key}
                  className={`finding-card${isHighlighted ? ' highlighted' : ''}`}
                  onClick={() => { if (!isEditing && !isDeleting) { onFindingSelect(f.track_id); setModalFinding(f) } }}
                >
                  <img
                    src={cropUrl(f)}
                    alt={`Finding ${f.track_id}`}
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                  />

                  <div className="finding-card-meta">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span
                        className="badge"
                        style={{ background: CATEGORY_COLOR[f.category ?? ''] ?? '#555' }}
                      >
                        {f.category ?? '?'}
                      </span>
                      <span
                        className="badge"
                        style={{ background: STATUS_COLOR[f.status] ?? '#555' }}
                      >
                        {f.status === 'validated' ? '✓' : f.status === 'rejected' ? '✗' : '!'}
                      </span>
                    </div>
                    <span style={{ fontSize: 10, color: '#777' }}>
                      {f.confidence != null ? `${(f.confidence * 100).toFixed(0)}% conf` : '—'}
                      {' · '}#{f.track_id}
                    </span>
                  </div>

                  {isEditing ? (
                    <div className="inline-edit" onClick={(e) => e.stopPropagation()}>
                      <select value={edit.category} onChange={(e) => setEdit((s) => ({ ...s, category: e.target.value }))}>
                        <option value="">— Kategorie —</option>
                        {LITTER_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <select value={edit.status} onChange={(e) => setEdit((s) => ({ ...s, status: e.target.value }))}>
                        <option value="">— Status —</option>
                        <option value="validated">validated</option>
                        <option value="rejected">rejected</option>
                        <option value="error">error</option>
                      </select>
                      <div className="inline-edit-actions">
                        <button className="save" onClick={() => void handleSave(f.mission_id, f.track_id)}>Speichern</button>
                        <button onClick={() => setEditingKey(null)}>Abbrechen</button>
                      </div>
                    </div>
                  ) : isDeleting ? (
                    <div className="inline-edit" onClick={(e) => e.stopPropagation()}>
                      <span style={{ fontSize: 10, color: '#f44336' }}>Wirklich löschen?</span>
                      <div className="inline-edit-actions">
                        <button className="save" style={{ color: '#f44336' }}
                          onClick={() => void handleDelete(f.mission_id, f.track_id)}>Ja</button>
                        <button onClick={() => setDeleteConfirm(null)}>Nein</button>
                      </div>
                    </div>
                  ) : (
                    <div className="card-actions">
                      <button className="card-btn" onClick={(e) => startEdit(f, e)}>Editieren</button>
                      <button
                        className="card-btn danger"
                        onClick={(e) => { e.stopPropagation(); setEditingKey(null); setDeleteConfirm(key) }}
                      >
                        Löschen
                      </button>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {modalFinding && (
        <div className="modal-overlay" onClick={() => setModalFinding(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: 15, color: '#ddd' }}>
                <span
                  className="badge"
                  style={{ background: CATEGORY_COLOR[modalFinding.category ?? ''] ?? '#555', marginRight: 8 }}
                >
                  {modalFinding.category ?? 'Unbekannt'}
                </span>
                Finding #{modalFinding.track_id}
              </h3>
              <button
                onClick={() => setModalFinding(null)}
                style={{ background: 'none', border: 'none', color: '#666', fontSize: 20, cursor: 'pointer', lineHeight: 1 }}
              >
                ×
              </button>
            </div>

            <div className="modal-images">
              <img src={cropUrl(modalFinding)} alt="Crop" />
              {modalFinding.context_image_path && (
                <img src={contextUrl(modalFinding)} alt="Kontext" />
              )}
            </div>

            <dl className="modal-meta">
              <dt>Status</dt>
              <dd>
                <span className="badge" style={{ background: STATUS_COLOR[modalFinding.status] }}>
                  {modalFinding.status}
                </span>
              </dd>

              <dt>Konfidenz</dt>
              <dd>{modalFinding.confidence != null ? `${(modalFinding.confidence * 100).toFixed(1)}%` : '—'}</dd>

              <dt>Modell</dt>
              <dd>{modalFinding.model_name ?? '—'}</dd>

              <dt>Zeitstempel</dt>
              <dd>{fmtNs(modalFinding.validated_at_ns)}</dd>

              <dt>Position</dt>
              <dd>
                {modalFinding.robot_x != null && modalFinding.robot_y != null
                  ? `x=${modalFinding.robot_x.toFixed(2)}, y=${modalFinding.robot_y.toFixed(2)}`
                  : '—'}
              </dd>

              <dt>Beobachtungen</dt>
              <dd>{modalFinding.n_observations}×</dd>

              <dt>Fläche</dt>
              <dd>{modalFinding.area_px} px²</dd>

              {modalFinding.description && (
                <>
                  <dt>Beschreibung</dt>
                  <dd style={{ gridColumn: 'span 1' }}>{modalFinding.description}</dd>
                </>
              )}
            </dl>
          </div>
        </div>
      )}
    </>
  )
}
