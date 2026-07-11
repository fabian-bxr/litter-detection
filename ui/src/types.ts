export interface Pose2D {
  x: number
  y: number
  theta: number
}

export interface FindingRow {
  mission_id: string
  track_id: number
  status: 'validated' | 'rejected' | 'error'
  category: string | null
  confidence: number | null
  description: string | null
  model_name: string | null
  robot_x: number | null
  robot_y: number | null
  bearing_rad: number
  bbox: [number, number, number, number]
  area_px: number
  n_observations: number
  validated_at_ns: number
  image_path: string | null
  context_image_path: string | null
}

export interface MissionRow {
  mission_id: string
  prompt: string | null
  started_ns: number | null
  finished_ns: number | null
  coverage_fraction: number | null
  distance_m: number | null
  n_waypoints: number | null
  n_blocked: number | null
  status_counts: Record<string, number>
}

export interface MapConfig {
  origin_x: number
  origin_y: number
  origin_theta: number
  resolution: number
  width_px: number | null
  height_px: number | null
}

export interface CoverageStats {
  fraction: number
  reachable_m2: number
}

export interface StateMessage {
  pose: Pose2D | null
  path_history: [number, number][]
  planned_path: [number, number][]
  nav_status: Record<string, unknown> | null
  overlay_seq?: number
  coverage?: CoverageStats | null
  error?: string
  keepalive?: boolean
}
