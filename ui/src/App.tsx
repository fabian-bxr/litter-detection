import { useEffect, useState } from 'react'
import CameraPanel from './components/CameraPanel'
import MapPanel from './components/MapPanel'
import MissionSelector from './components/MissionSelector'
import FindingsGallery from './components/FindingsGallery'
import ChatPanel from './components/ChatPanel'
import type { FindingRow, MissionRow } from './types'
import './App.css'

/** Poll interval for the findings of a mission that is currently running. */
const LIVE_FINDINGS_POLL_MS = 3000

export default function App() {
  const [missions, setMissions] = useState<MissionRow[]>([])
  const [selectedMissionId, setSelectedMissionId] = useState<string | null>(null)
  const [findings, setFindings] = useState<FindingRow[]>([])
  const [highlightedTrackId, setHighlightedTrackId] = useState<number | null>(null)
  const [runningMissionId, setRunningMissionId] = useState<string | null>(null)

  // ── Load missions list ────────────────────────────────────────────────────
  function loadMissions(autoSelect = false) {
    fetch('/api/missions')
      .then((r) => r.json() as Promise<MissionRow[]>)
      .then((data) => {
        setMissions(data)
        if (autoSelect && data.length > 0) {
          setSelectedMissionId(data[0].mission_id)
        }
      })
      .catch(console.error)
  }

  function loadFindings(missionId: string) {
    fetch(`/api/missions/${missionId}/findings`)
      .then((r) => r.json() as Promise<FindingRow[]>)
      .then(setFindings)
      .catch(console.error)
  }

  useEffect(() => { loadMissions(true) }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load findings when mission changes ────────────────────────────────────
  useEffect(() => {
    if (!selectedMissionId) { setFindings([]); return }
    loadFindings(selectedMissionId)
  }, [selectedMissionId])

  // ── Stream in findings while the selected mission is still running ────────
  // The mission is validating detections as it walks; without this the board
  // would sit empty until the run ends.
  useEffect(() => {
    if (!runningMissionId || runningMissionId !== selectedMissionId) return
    const id = setInterval(() => loadFindings(runningMissionId), LIVE_FINDINGS_POLL_MS)
    return () => clearInterval(id)
  }, [runningMissionId, selectedMissionId])

  // ── Mutation callbacks for gallery ────────────────────────────────────────
  function handleDelete(missionId: string, trackId: number) {
    setFindings((prev) => prev.filter((f) => !(f.mission_id === missionId && f.track_id === trackId)))
    if (highlightedTrackId === trackId) setHighlightedTrackId(null)
  }

  function handleUpdate(updated: FindingRow) {
    setFindings((prev) => prev.map((f) =>
      f.mission_id === updated.mission_id && f.track_id === updated.track_id ? updated : f,
    ))
  }

  // ── Called by ChatPanel when a mission starts ─────────────────────────────
  // The backend has already created the board for `missionId`, so switch to it
  // at once: the previous run's findings clear instead of lingering on the map
  // and in the gallery for the whole run.
  function handleMissionStart(missionId: string) {
    setFindings([])
    setHighlightedTrackId(null)
    setSelectedMissionId(missionId)
    setRunningMissionId(missionId)
    loadMissions()  // the new (still empty) mission joins the selector
  }

  // ── Called by ChatPanel when a mission finishes ───────────────────────────
  function handleMissionComplete() {
    setRunningMissionId(null)
    loadMissions()  // refresh the selector's finding counts / coverage
    // The selection already points at the finished mission, so the
    // selectedMissionId effect won't refire — pull its final findings here.
    if (selectedMissionId) loadFindings(selectedMissionId)
  }

  return (
    <div className="app">
      {/* ── Toolbar ── */}
      <div className="toolbar">
        <MissionSelector
          missions={missions}
          selectedId={selectedMissionId}
          onSelect={setSelectedMissionId}
        />
      </div>

      {/* ── Main panels ── */}
      <div className="top-panels">
        <div className="panel">
          <div className="panel-label">Kamera</div>
          <div className="panel-body">
            <CameraPanel />
          </div>
        </div>

        <div className="panel">
          <div className="panel-label">Karte &amp; Pfadplanung</div>
          <div className="panel-body">
            <MapPanel
              findings={findings}
              onFindingSelect={setHighlightedTrackId}
              highlightedTrackId={highlightedTrackId}
            />
          </div>
        </div>

        <div className="panel">
          <div className="panel-label">Mission</div>
          <div className="panel-body">
            <ChatPanel
            onMissionStart={handleMissionStart}
            onMissionComplete={handleMissionComplete}
          />
          </div>
        </div>
      </div>

      {/* ── Findings gallery ── */}
      <FindingsGallery
        missionId={selectedMissionId}
        findings={findings}
        onFindingSelect={setHighlightedTrackId}
        highlightedTrackId={highlightedTrackId}
        onDelete={handleDelete}
        onUpdate={handleUpdate}
      />
    </div>
  )
}
