import React, { useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

type RecordData = Record<string, unknown>
type Metric = { update?: number; loss?: number; td_error?: number; mean_q?: number; [key: string]: unknown }
type Training = {
  id?: string; input_version?: number; output_version?: number; updates_completed?: number
  total_updates?: number; metrics?: Metric[]; probe_before?: unknown; probe_after?: unknown
  elapsed_seconds?: number; status?: string; error?: string; [key: string]: unknown
}
type Match = {
  id?: string; index?: number; policy_version?: number; epsilon?: number; elapsed_seconds?: number
  remaining_seconds?: number; human_return?: number; machine_return?: number; decisions?: number
  score_mode?: string; [key: string]: unknown
}
type Snapshot = {
  id?: string; status: string; active_version?: number; config?: RecordData; replay_size?: number
  match?: Match | null; training?: Training | null; last_result?: RecordData | null
  comparison?: RecordData | null; history?: RecordData[]; error?: string | null; events_seq?: number
  [key: string]: unknown
}
type SavedSession = { id: string; status?: string; active_version?: number; [key: string]: unknown }

const EMPTY: Snapshot = { status: 'no_session' }
const numberOf = (...values: unknown[]): number | undefined => {
  for (const value of values) if (typeof value === 'number' && Number.isFinite(value)) return value
  return undefined
}
const textOf = (...values: unknown[]): string | undefined => {
  for (const value of values) if (typeof value === 'string' && value.length) return value
  return undefined
}
const fmt = (value: unknown, digits = 2) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—'
const version = (value: unknown) => typeof value === 'number' ? `v${value}` : '—'
const signed = (value: number | undefined) => value === undefined ? '—' : `${value > 0 ? '+' : ''}${fmt(value)}`
const recordOf = (value: unknown): RecordData => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as RecordData : {}
const duration = (seconds: number | undefined) => seconds === undefined ? '—' : `${Math.floor(Math.max(0, seconds) / 60).toString().padStart(2, '0')}:${Math.floor(Math.max(0, seconds) % 60).toString().padStart(2, '0')}`
const labels: Record<string, string> = {
  no_session: 'Not started', ready: 'Ready to play', playing: 'Match in progress', learning: 'Learning',
  summary: 'Learning complete', insufficient_data: 'Collecting experience', failed: 'Needs attention',
  cancelled: 'Learning cancelled', interrupted: 'Session interrupted', restoring: 'Restoring session',
}

async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = payload.detail ?? payload.error ?? `${response.status} ${response.statusText}`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return payload as T
}

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    play: <path d="m9 5 11 7-11 7Z" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
    arrow: <><path d="M5 12h14m-5-5 5 5-5 5" /></>,
    chart: <><path d="M4 4v16h17M8 14l4-5 4 3 5-7" /></>,
    history: <><path d="M3 11a9 9 0 1 1 2 7M3 5v6h6m3-4v6l4 2" /></>,
    game: <><path d="M7 8h10a4 4 0 0 1 4 4l1 4a3 3 0 0 1-5 2l-2-2H9l-2 2a3 3 0 0 1-5-2l1-4a4 4 0 0 1 4-4Z" /><path d="M7 10v5m-2-2h4m7-1h.01m3 2h.01" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    brain: <><path d="M12 3v18M12 6C8 0 2 5 5 9c-5 4-1 11 4 9 0 3 3 3 3 0m0-12c4-6 10-1 7 3 5 4 1 11-4 9 0 3-3 3-3 0" /><path d="m6 8 3 2m9-2-3 2M7 15l2-2m8 2-2-2" /></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5" /></>,
    refresh: <><path d="M20 11a8 8 0 0 0-14-5L3 9m0-6v6h6m-5 4a8 8 0 0 0 14 5l3-3m0 6v-6h-6" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10h.01" /></>,
    check: <path d="m5 12 4 4L20 5" />,
    plus: <path d="M12 5v14M5 12h14" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.info}</svg>
}

function LineChart({ metrics, field, color, label }: { metrics: Metric[]; field: string; color: string; label: string }) {
  const samples = metrics.map((point, index) => ({ x: numberOf(point.update) ?? index + 1, y: numberOf(point[field]) })).filter((point): point is { x: number; y: number } => point.y !== undefined)
  const width = 430, height = 150, top = 14, bottom = 24, left = 49, right = 9
  if (!samples.length) return <div className="chart-empty"><Icon name="chart" size={28} /><span>Actual {label.toLowerCase()} will appear here</span><small>Updates begin after a completed match</small></div>
  let min = Math.min(...samples.map(p => p.y)), max = Math.max(...samples.map(p => p.y))
  const pad = (max - min) * 0.14 || Math.max(Math.abs(max) * 0.1, 0.01)
  min -= pad; max += pad
  const first = samples[0].x, last = samples[samples.length - 1].x
  const points = samples.map(p => [left + (p.x - first) / Math.max(last - first, 1) * (width - left - right), top + (max - p.y) / (max - min) * (height - top - bottom)])
  const line = points.map(p => p.join(',')).join(' ')
  return <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} over ${samples.length} recorded updates`}>
    {[0, 0.5, 1].map((fraction, i) => <g key={i}>
      <line x1={left} y1={top + fraction * (height - top - bottom)} x2={width-right} y2={top + fraction * (height - top - bottom)} stroke="#e5e8e2" strokeDasharray="3 4" />
      <text x={left - 8} y={top + fraction * (height - top - bottom) + 4} textAnchor="end">{fmt(max - fraction * (max-min), 3)}</text>
    </g>)}
    <polygon points={`${left},${height-bottom} ${line} ${points[points.length-1][0]},${height-bottom}`} fill={color} opacity="0.07" />
    <polyline points={line} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    <circle cx={points[points.length-1][0]} cy={points[points.length-1][1]} r="3" fill={color} />
    <text x={left} y={height-5}>Update {first}</text><text x={width-right} y={height-5} textAnchor="end">{last}</text>
  </svg>
}

function GameFeed({ playing, hasMatch }: { playing: boolean; hasMatch: boolean }) {
  const [src, setSrc] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const active = useRef(playing)
  active.current = playing
  const refresh = useCallback(() => setSrc(`/api/frame.jpg?t=${Date.now()}`), [])
  useEffect(() => {
    if (playing || hasMatch) refresh()
    return () => { if (timer.current) clearTimeout(timer.current) }
  }, [playing, hasMatch, refresh])
  const next = () => { if (active.current) timer.current = setTimeout(refresh, 66) }
  return <div className="game-feed">
    {src && <img className={loaded ? 'arcade-frame loaded' : 'arcade-frame'} src={src} alt="Live Atari Pong game" onLoad={() => { setLoaded(true); next() }} onError={() => { if (!loaded) setSrc(null); if (active.current) timer.current = setTimeout(refresh, 400) }} />}
    {!loaded && <div className="game-empty">
      <div className="paddle-symbol" aria-hidden="true"><span /><i /><span /></div>
      <strong>{playing ? 'Connecting to Atari…' : 'Your next opponent learns.'}</strong>
      <p>{playing ? 'Waiting for the first frame from the game engine.' : 'Play a match. Watch the network train. Meet the next version.'}</p>
      <span className="engine-tag">ATARI PONG · REAL ENVIRONMENT</span>
    </div>}
  </div>
}

function App() {
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY)
  const [online, setOnline] = useState<boolean | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [saved, setSaved] = useState<SavedSession[]>([])
  const [selectedSession, setSelectedSession] = useState('')
  const [durationSetting, setDurationSetting] = useState(120)
  const [updatesSetting, setUpdatesSetting] = useState(450)
  const [seed, setSeed] = useState(7)
  const [setupOpen, setSetupOpen] = useState(false)
  const [chartField, setChartField] = useState('loss')
  const [expandedRow, setExpandedRow] = useState<string | null>(null)
  const socket = useRef<WebSocket | null>(null)
  const keys = useRef(new Set<string>())
  const gameRegion = useRef<HTMLDivElement>(null)
  const status = snapshot.status
  const playing = status === 'playing', learning = status === 'learning'
  const hasSession = Boolean(snapshot.id)
  const currentMatch = snapshot.match ?? recordOf(snapshot.last_result) as Match
  const training = snapshot.training
  const metrics = Array.isArray(training?.metrics) ? training.metrics : []
  const latest = metrics.at(-1)
  const history = Array.isArray(snapshot.history) ? snapshot.history : []
  const config = snapshot.config ?? {}
  const last = recordOf(snapshot.last_result)
  const completedHistory = history.filter(row => row.valid !== false && row.status !== 'aborted' && row.status !== 'invalid' && row.reason !== 'aborted')
  const previous = completedHistory.length > 1 ? completedHistory.at(-2) : undefined
  const matchIndex = numberOf(snapshot.match?.index, last.index, last.match_index) ?? history.length + (playing ? 1 : 0)
  const humanReturn = numberOf(currentMatch?.human_return, last.human_return)
  const machineReturn = numberOf(currentMatch?.machine_return, last.machine_return)
  const updates = numberOf(training?.updates_completed) ?? 0
  const totalUpdates = numberOf(training?.total_updates, config.updates_per_match) ?? 450
  const replay = numberOf(snapshot.replay_size) ?? 0
  const minReplay = numberOf(config.min_replay) ?? 256
  const canPlay = hasSession && !playing && !learning && !busy

  const refreshSaved = useCallback(async () => {
    try {
      const value = await api<SavedSession[] | { sessions: SavedSession[] }>('/api/sessions')
      setSaved(Array.isArray(value) ? value : value.sessions ?? [])
    } catch { /* Session discovery is retried after the backend reconnects. */ }
  }, [])
  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const value = await api<Snapshot>('/api/session')
        if (alive) { setSnapshot(value); setOnline(true) }
      } catch { if (alive) setOnline(false) }
    }
    void poll(); void refreshSaved()
    const interval = setInterval(poll, 600)
    const source = new EventSource('/api/events')
    source.onmessage = event => {
      try { const value = JSON.parse(event.data); if (value.status && alive) { setSnapshot(value); setOnline(true) } } catch { /* Ignore a partial reconnect event. */ }
    }
    return () => { alive = false; clearInterval(interval); source.close() }
  }, [refreshSaved])

  useEffect(() => {
    if (!playing) { keys.current.clear(); return }
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/play`)
    socket.current = ws
    let blurred = false
    const send = () => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ keys: [...keys.current] }))
    }
    const release = () => {
      keys.current.clear(); blurred = true
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ keys: [], blur: true }))
    }
    ws.onopen = () => { blurred = false; send(); gameRegion.current?.focus({ preventScroll: true }) }
    const keyName = (key: string) => ({ ArrowUp: 'up', ArrowDown: 'down', w: 'up', W: 'up', s: 'down', S: 'down', ' ': 'fire' })[key]
    const down = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.matches('input, select, textarea')) return
      const key = keyName(event.key)
      if (key) { event.preventDefault(); keys.current.add(key); send() }
      if (event.key === 'Escape') { event.preventDefault(); void perform('/api/matches/abort', {}, 'Stopping match') }
    }
    const up = (event: KeyboardEvent) => { const key = keyName(event.key); if (key) { event.preventDefault(); keys.current.delete(key); send() } }
    const visibility = () => { if (document.hidden) release() }
    window.addEventListener('keydown', down); window.addEventListener('keyup', up)
    window.addEventListener('blur', release); document.addEventListener('visibilitychange', visibility)
    const heartbeat = setInterval(() => { if (!blurred) send() }, 100)
    return () => {
      clearInterval(heartbeat); keys.current.clear()
      window.removeEventListener('keydown', down); window.removeEventListener('keyup', up)
      window.removeEventListener('blur', release); document.removeEventListener('visibilitychange', visibility)
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ keys: [] }))
      ws.close(); if (socket.current === ws) socket.current = null
    }
  // A websocket belongs to a match, not every streamed snapshot.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, snapshot.match?.id])

  const perform = async (url: string, body: unknown, description: string) => {
    setBusy(description); setError(null)
    try {
      const value = await api<Snapshot>(url, body)
      setSnapshot(value); setOnline(true); setSetupOpen(false)
      if (url === '/api/sessions') void refreshSaved()
    } catch (err) { setError(err instanceof Error ? err.message : String(err)) }
    finally { setBusy(null) }
  }
  const newSession = () => void perform('/api/sessions', { config: {
    match_seconds: durationSetting, updates_per_match: updatesSetting, min_replay: 256,
    batch_size: 32, replay_capacity: 20000, seed, device: 'cpu',
  } }, 'Preparing the model')
  const press = (key: string, held: boolean) => {
    if (held) keys.current.add(key); else keys.current.delete(key)
    const ws = socket.current
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ keys: [...keys.current] }))
  }
  const exportHistory = () => {
    const fields = ['index', 'id', 'policy_version', 'human_return', 'machine_return', 'elapsed_seconds', 'status', 'reason']
    const escape = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`
    const content = [fields.join(','), ...history.map(row => fields.map(field => escape(row[field])).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8;' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `pong-${snapshot.id}-history.csv`; anchor.click(); URL.revokeObjectURL(url)
  }
  const stage = playing ? 0 : learning ? 2 : training && status !== 'insufficient_data' ? 3 : history.length ? 1 : 0
  const machineDelta = numberOf(last.machine_return) !== undefined && numberOf(previous?.machine_return) !== undefined
    ? (last.machine_return as number) - (previous!.machine_return as number) : undefined
  const humanDelta = numberOf(last.human_return) !== undefined && numberOf(previous?.human_return) !== undefined
    ? (last.human_return as number) - (previous!.human_return as number) : undefined
  const probeBefore = recordOf(training?.probe_before), probeAfter = recordOf(training?.probe_after)

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#top" aria-label="Pong Learning Lab home"><span className="brand-mark"><i /><i /><i /><i /></span><span>PONG<span className="brand-sub">LEARNING LAB</span></span></a>
      <span className="side-label">WORKSPACE</span>
      <a className="nav-item active" href="#arena"><Icon name="game" />Play & learn<span className="nav-dot" /></a>
      <a className="nav-item" href="#learning"><Icon name="chart" />Learning monitor</a>
      <a className="nav-item" href="#history"><Icon name="history" />Match history</a>
      <div className="side-session"><span className="side-label">ACTIVE SESSION</span><strong>{hasSession ? snapshot.id?.slice(0, 20) : 'Your lab is ready'}</strong><p>{hasSession ? 'Experience carries into every next game.' : 'Create a session to begin learning from play.'}</p><button className="side-button" onClick={() => setSetupOpen(!setupOpen)} disabled={playing || learning}><Icon name="plus" size={16} />New / restore session</button></div>
      <div className="side-bottom"><span className="network-dot" />LOCAL EXPERIMENT<div>DQN · Atari Pong</div><p>One game. One learning step.<br />A different opponent next time.</p></div>
    </aside>

    <main id="top">
      <header className="page-header"><div><div className="eyebrow">REINFORCEMENT LEARNING / INTERACTIVE DEMO</div><h1>Play. Learn. Play again<span>.</span></h1><p>You play the game. The network learns from the experience.</p></div><div className={`connection ${online === false ? 'offline' : ''}`}><span />{online === null ? 'Connecting' : online ? 'Local engine connected' : 'Engine disconnected'}</div></header>

      {(error || snapshot.error || online === false) && <div className="error-banner" role="alert"><Icon name="info" /><div><strong>{online === false ? 'The local engine is not connected.' : 'The experiment needs attention.'}</strong><p>{error ?? snapshot.error ?? 'Start the Pong backend, then this page will reconnect automatically. No game data is being simulated.'}</p></div>{error && <button aria-label="Dismiss error" onClick={() => setError(null)}><Icon name="close" size={17} /></button>}</div>}

      {(!hasSession || setupOpen) && <section className="setup-card" aria-labelledby="setup-title"><div className="setup-intro"><span className="pill neutral">START HERE</span><h2 id="setup-title">Meet your first opponent.</h2><p>A fresh DQN starts with random weights. Each completed game adds experience; learning happens after the final whistle.</p></div><div className="setup-fields"><label>Match length<select value={durationSetting} onChange={e => setDurationSetting(Number(e.target.value))}><option value={120}>2 minutes · full demo</option><option value={20}>20 seconds · quick cycle</option><option value={60}>1 minute</option></select></label><label>Updates after a game<select value={updatesSetting} onChange={e => setUpdatesSetting(Number(e.target.value))}><option value={450}>450 updates · recommended</option><option value={128}>128 updates</option><option value={32}>32 updates · quick cycle</option></select></label><label className="seed-label">Session seed<input type="number" min="0" max="4294967294" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label><button className="button primary create-button" onClick={newSession} disabled={Boolean(busy) || online === false}><Icon name="plus" size={17} />{busy ?? 'Create session'}</button>{saved.length > 0 && <div className="restore-row"><select aria-label="Saved session" value={selectedSession} onChange={e => setSelectedSession(e.target.value)}><option value="">Restore a saved session…</option>{saved.map(item => <option value={item.id} key={item.id}>{item.id} {item.active_version !== undefined ? `· v${item.active_version}` : ''}</option>)}</select><button className="button secondary" disabled={!selectedSession || Boolean(busy)} onClick={() => void perform('/api/sessions', { session_id: selectedSession }, 'Restoring session')}><Icon name="refresh" size={15} />Restore</button></div>}</div></section>}

      <div className="journey" aria-label="Experiment cycle">{['Play a match', 'Review the result', 'Train the network', 'Meet the next version'].map((step, index) => <React.Fragment key={step}><div className={index === stage ? 'journey-step current' : 'journey-step'}><span>{String(index+1).padStart(2, '0')}</span>{step}</div>{index < 3 && <span className="journey-connector" />}</React.Fragment>)}</div>

      <div className="workspace-grid">
        <section className="arena-card" id="arena" aria-labelledby="arena-title"><div className="card-heading"><div><span className="section-index">01 / THE ARENA</span><h2 id="arena-title">Human vs. Deep Q-Network</h2></div><span className={`pill ${playing || learning ? 'live' : 'neutral'}`}><span className="status-dot" />{labels[status] ?? status.replaceAll('_', ' ')}</span></div>
          <div className="scoreboard"><div className="player human"><span className="player-label"><span className="player-icon">H</span>YOU<span className="player-side">HUMAN</span></span><strong>{fmt(humanReturn, 0)}</strong></div><div className="match-clock"><span>{matchIndex ? `MATCH ${String(matchIndex).padStart(2, '0')}` : 'NEXT MATCH'}</span><strong>{duration(playing ? numberOf(snapshot.match?.remaining_seconds) : numberOf(config.match_seconds, durationSetting))}</strong><small>SIMULATION TIME</small></div><div className="player machine"><span className="player-label"><span className="player-side">DQN {version(snapshot.match?.policy_version ?? snapshot.active_version)}</span>MACHINE<span className="player-icon">Q</span></span><strong>{fmt(machineReturn, 0)}</strong></div></div>
          <div className="game-region" tabIndex={0} ref={gameRegion} aria-label="Pong game. Use up and down arrows or W and S to move, and space to serve."><GameFeed playing={playing} hasMatch={Boolean(snapshot.match || snapshot.last_result)} /><div className="feed-bottom"><span><span className={`live-light ${playing ? 'on' : ''}`} />{playing ? 'LIVE ATARI' : learning ? 'MATCH COMPLETE · WEIGHTS UPDATING' : hasSession ? 'READY FOR THE NEXT MATCH' : 'WAITING FOR A SESSION'}</span><span>84 × 84 × 4 INPUT</span></div></div>
          <div className="arena-actions"><div className="keyboard-help"><span className="key">↑</span><span className="key">↓</span><span className="or">or W / S</span><span className="key space">SPACE</span><span>serve</span></div>{playing ? <button className="button stop" disabled={Boolean(busy)} onClick={() => void perform('/api/matches/abort', {}, 'Stopping match')}><Icon name="stop" size={16} />Stop match</button> : learning ? <button className="button secondary" disabled={Boolean(busy)} onClick={() => void perform('/api/training/cancel', {}, 'Cancelling updates')}><Icon name="stop" size={15} />Cancel learning</button> : <button className="button primary" disabled={!canPlay || online === false} onClick={() => void perform('/api/matches', {}, 'Starting Atari')}><Icon name="play" size={17} />{busy ?? (history.length ? 'Play again' : 'Start match')}<Icon name="arrow" size={17} /></button>}</div>
          {playing && <div className="touch-controls" aria-label="Touch controls">{[['up', '↑ Move up'], ['down', '↓ Move down'], ['fire', 'Serve']].map(([key, title]) => <button key={key} onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); press(key, true) }} onPointerUp={() => press(key, false)} onPointerCancel={() => press(key, false)}>{title}</button>)}</div>}
          <div className="arena-footnote"><Icon name="info" size={14} /><span>Scores show environment return, including serve penalties. Model weights stay fixed during a match.</span></div>
        </section>

        <section className="learning-card" id="learning" aria-labelledby="learning-title"><div className="card-heading"><div><span className="section-index">02 / UNDER THE HOOD</span><h2 id="learning-title">The learning monitor</h2></div><span className="brain-icon"><Icon name="brain" size={24} /></span></div>
          <div className="model-banner"><span>ACTIVE MODEL</span><strong>{version(snapshot.active_version)}</strong><small>{snapshot.active_version === 0 ? 'Starting from scratch' : hasSession ? 'Retained across games' : 'Create a session to initialize'}</small></div>
          <div className="learning-stats"><div><span>Experience replay</span><strong>{fmt(replay, 0)}<small> transitions</small></strong></div><div><span>Exploration ε</span><strong>{fmt(numberOf(snapshot.match?.epsilon, snapshot.epsilon, last.epsilon), 3)}</strong></div></div>
          <div className="training-state"><div><span className={`tiny-dot ${learning ? 'pulse' : ''}`} /><strong>{learning ? 'Updating the Q-network' : status === 'insufficient_data' ? 'A little more experience first' : training ? labels[status] ?? 'Last learning phase' : 'Learning begins after the game'}</strong></div><p>{learning ? `Sampling from ${fmt(replay, 0)} retained transitions.` : status === 'insufficient_data' ? `${fmt(replay, 0)} / ${fmt(minReplay, 0)} sampleable transitions. Keep playing to fill replay.` : training ? `Input ${version(training.input_version)} → ${training.output_version === undefined || training.output_version === null ? 'no new version' : version(training.output_version)}. Actual progress is shown below.` : 'No gradients during play. Completed matches become the next training experience.'}</p><div className="progress-track"><span style={{ width: `${Math.min(100, updates / Math.max(totalUpdates, 1) * 100)}%` }} /></div><div className="progress-label"><span>{fmt(updates, 0)} / {fmt(totalUpdates, 0)} updates</span><span>{training ? `${fmt(training.elapsed_seconds, 1)}s` : 'Not started'}</span></div></div>
          <div className="chart-tabs" aria-label="Learning chart">{[['loss', 'Loss'], ['td_error', 'TD error'], ['mean_q', 'Mean Q']].map(([field, label]) => <button className={chartField === field ? 'selected' : ''} key={field} onClick={() => setChartField(field)}>{label}</button>)}<strong>{fmt(latest?.[chartField], 4)}</strong></div>
          <LineChart metrics={metrics} field={chartField} color="#238b65" label={chartField === 'loss' ? 'Huber loss' : chartField === 'td_error' ? 'Absolute TD error' : 'Mean Q-value'} />
          <div className="probe-box"><div><Icon name="grid" size={16} /><strong>Same-sample check</strong><span>BEFORE → AFTER</span></div><p>{training?.probe_before && training?.probe_after ? <>Mean Q <b>{fmt(numberOf(probeBefore.mean_q, probeBefore.q_mean), 4)}</b> → <b>{fmt(numberOf(probeAfter.mean_q, probeAfter.q_mean), 4)}</b><br />TD error <b>{fmt(numberOf(probeBefore.td_error, probeBefore.mean_td_error), 4)}</b> → <b>{fmt(numberOf(probeAfter.td_error, probeAfter.mean_td_error), 4)}</b></> : 'A fixed replay sample will show how predictions changed after learning.'}</p><small>Training diagnostics, not a gameplay evaluation.</small></div>
          {(status === 'failed' || status === 'cancelled' || status === 'interrupted') && training && <button className="button secondary retry-button" disabled={Boolean(busy)} onClick={() => void perform('/api/training/retry', {}, 'Retrying learning')}><Icon name="refresh" size={16} />Retry uncommitted learning</button>}
        </section>
      </div>

      <section className="comparison-card" aria-labelledby="comparison-title"><div className="comparison-intro"><span className="section-index">03 / WHAT CHANGED?</span><h2 id="comparison-title">One game at a time.</h2><p>{previous ? 'The latest completed match, compared with the previous valid game.' : 'Complete two valid matches to make your first comparison.'}</p></div><div className="delta-stat"><span>MACHINE RETURN</span><strong>{signed(machineDelta)}</strong><small>{previous ? `Previous ${fmt(previous.machine_return)} → latest ${fmt(last.machine_return)}` : 'No previous game yet'}</small></div><div className="delta-stat"><span>HUMAN RETURN</span><strong>{signed(humanDelta)}</strong><small>{previous ? `Previous ${fmt(previous.human_return)} → latest ${fmt(last.human_return)}` : 'Your first game starts the record'}</small></div><div className="comparison-note"><Icon name="info" size={17} /><p>These are observed match results. You and the exploration policy can change too; one better score does not prove a better model.</p></div></section>

      <section className="history-card" id="history" aria-labelledby="history-title"><div className="card-heading"><div><span className="section-index">04 / THE EXPERIMENT LOG</span><h2 id="history-title">Session history<span className="count-badge">{history.length}</span></h2></div><button className="text-button" onClick={exportHistory} disabled={!history.length}><Icon name="download" size={16} />Export CSV</button></div>{history.length ? <div className="table-scroll"><table><thead><tr><th>MATCH</th><th>MODEL USED</th><th>YOU</th><th>MACHINE</th><th>DURATION</th><th>RESULT</th><th /></tr></thead><tbody>{[...history].reverse().map((row, index) => {
        const rowId = textOf(row.id, row.match_id) ?? String(index)
        const valid = row.valid !== false && row.status !== 'aborted' && row.status !== 'invalid' && row.reason !== 'aborted'
        const human = numberOf(row.human_return), machine = numberOf(row.machine_return)
        const result = textOf(row.outcome, row.result) ?? (!valid ? textOf(row.reason, row.status) ?? 'Excluded' : human === undefined || machine === undefined ? 'Recorded' : human === machine ? 'Draw' : machine > human ? 'Machine leads' : 'You lead')
        return <React.Fragment key={rowId}><tr><td><span className="match-number">{String(numberOf(row.index, row.match_index) ?? history.length-index).padStart(2, '0')}</span><small>{rowId.slice(0, 18)}</small></td><td><span className="model-chip">{version(row.policy_version)}</span></td><td>{fmt(human)}</td><td>{fmt(machine)}</td><td>{duration(numberOf(row.elapsed_seconds, row.duration_seconds))}</td><td><span className={`result-pill ${!valid ? 'excluded' : machine !== undefined && human !== undefined && machine > human ? 'machine-won' : ''}`}>{result}</span></td><td><button className="row-detail" aria-label={`Show details for match ${rowId}`} onClick={() => setExpandedRow(expandedRow === rowId ? null : rowId)}>{expandedRow === rowId ? '−' : '+'}</button></td></tr>{expandedRow === rowId && <tr className="details-row"><td colSpan={7}><pre>{JSON.stringify(row, null, 2)}</pre></td></tr>}</React.Fragment>
      })}</tbody></table></div> : <div className="history-empty"><span><Icon name="history" size={24} /></span><div><strong>A clean slate.</strong><p>Your matches, model versions and learning cycles will be recorded here.</p></div></div>}</section>
      <footer><span>PONG LEARNING LAB<span className="footer-sep">/</span>AN EXPERIMENT IN DEEP Q-LEARNING</span><span>Real gameplay. Real updates. No annotated data.</span></footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
