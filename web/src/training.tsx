import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Icon, api, fmt } from './shared'

/* ------------------------------------------------------------------------------
 * "How they learned": an interactive, no-maths-needed explainer of Deep Q-Learning
 * built around the lab's real trained models and their real training curves.
 * ---------------------------------------------------------------------------- */

type Point = [number, number]
type Curves = {
  window: number
  pong_single: { training: Record<string, Point[]>; validation: Record<string, Point[]> }
  pong_two_player: { training: Record<string, Point[]>; evaluation: Record<string, Point[]> }
  breakout: { training: Record<string, Point[]> }
}
type Series = { id: string; label: string; color: string; points: Point[]; dots?: boolean }

// Validated categorical slots (blue, orange, aqua) on the card surface #fffefb.
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
const INK = { primary: '#1b2c23', secondary: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7' }
const steps = (value: number) => value >= 1e6 ? `${fmt(value / 1e6, value % 1e6 ? 1 : 0)}M` : value >= 1e3 ? `${fmt(value / 1e3, 0)}k` : fmt(value, 0)

/* ---------- glossary ---------------------------------------------------------- */
const GLOSSARY: Record<string, string> = {
  'Q-value': 'The network’s guess of how many points it will win from now on if it presses this button. Higher is better.',
  'decision': 'One choice of button. The agent decides 15 times a second; each choice is held for 4 Atari frames.',
  'reward': 'The only feedback it gets: +1 when it wins a point (or breaks a brick), −1 when it loses one, 0 otherwise.',
  'episode': 'One complete game from start to game over. (In the upstream Breakout run, losing one life ends an episode.)',
  'ε (epsilon)': 'The chance of pressing a random button instead of the best one. High ε = exploring; low ε = using what it knows.',
  'replay memory': 'A big notebook of past moments (screen, button, reward, next screen). Learning picks random pages from it.',
  'target network': 'A frozen copy of the network used to make the “what should have happened” answer. It is refreshed now and then so learning stays stable.',
  'evaluation': 'A test: the agent plays a few games with exploration (almost) off and we record its average score.',
}
function Term({ word, children }: { word: keyof typeof GLOSSARY | string; children?: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return <span className="tx-term-wrap" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
    <button type="button" className="tx-term" aria-expanded={open} onClick={() => setOpen(!open)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>{children ?? word}</button>
    {open && <span className="tx-pop" role="tooltip"><b>{word}</b>{GLOSSARY[word]}</span>}
  </span>
}

/* ---------- reusable line chart with crosshair, legend, end labels, table ------- */
function LineChart({ series, height = 290, yLabel, xLabel = 'Training decisions', markers = [], yDomain, caption }: {
  series: Series[]; height?: number; yLabel: string; xLabel?: string; caption: string
  markers?: { x: number; label: string }[]; yDomain?: [number, number]
}) {
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  // One series needs no legend or end label (the caption names it); give the plot that room.
  const single = series.length === 1
  const W = 720, H = height, L = 52, R = single ? 24 : 150, T = 16, B = 40
  const all = series.flatMap(s => s.points)
  if (!all.length) return <div className="tx-chart-empty">Loading the training record…</div>
  const xMin = Math.min(...all.map(p => p[0])), xMax = Math.max(...all.map(p => p[0]))
  const dataMin = Math.min(...all.map(p => p[1])), dataMax = Math.max(...all.map(p => p[1]))
  // A preferred range is widened if the data goes beyond it, so no line is ever drawn off the chart.
  let [yMin, yMax] = yDomain ? [Math.min(yDomain[0], dataMin), Math.max(yDomain[1], dataMax)] : [dataMin, dataMax]
  if (!yDomain || dataMin < yDomain[0] || dataMax > yDomain[1]) { const pad = (yMax - yMin) * 0.06 || 1; if (!yDomain || dataMin < yDomain[0]) yMin -= pad; if (!yDomain || dataMax > yDomain[1]) yMax += pad }
  const x = (v: number) => L + (v - xMin) / Math.max(xMax - xMin, 1) * (W - L - R)
  const y = (v: number) => T + (yMax - v) / Math.max(yMax - yMin, 1e-9) * (H - T - B)
  const ticks = (lo: number, hi: number, n: number) => {
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(m => m * mag).find(m => m >= raw) ?? raw
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out
  }
  const nearest = (points: Point[], at: number) => {
    let lo = 0, hi = points.length - 1
    while (lo < hi) { const mid = (lo + hi) >> 1; if (points[mid][0] < at) lo = mid + 1; else hi = mid }
    const a = points[Math.max(0, lo - 1)], b = points[lo]
    return a && Math.abs(a[0] - at) < Math.abs(b[0] - at) ? a : b
  }
  // End labels, nudged apart so they never overlap.
  const ends = (single ? [] : series).map(s => ({ s, last: s.points[s.points.length - 1], y: y(s.points[s.points.length - 1][1]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 30) ends[i].y = ends[i - 1].y + 30
  const move = (clientX: number) => {
    const rect = box.current?.querySelector('svg')?.getBoundingClientRect(); if (!rect) return
    const px = (clientX - rect.left) / rect.width * W
    setHoverX(Math.min(xMax, Math.max(xMin, xMin + (px - L) / (W - L - R) * (xMax - xMin))))
  }
  const key = (event: React.KeyboardEvent) => {
    const stride = (xMax - xMin) / 50
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); setHoverX(current => Math.min(xMax, Math.max(xMin, (current ?? xMax) + (event.key === 'ArrowRight' ? stride : -stride))))
    }
  }
  const readout = hoverX === null ? null : series.map(s => ({ s, p: nearest(s.points, hoverX) }))
  const tableRows = Array.from({ length: 11 }, (_, i) => xMin + (xMax - xMin) * i / 10)
  return <figure className="tx-figure">
    <div className="tx-chart-top">
      <div className="tx-legend">{!single && series.map(s => <span key={s.id}><i style={{ background: s.color }} />{s.label}</span>)}</div>
      <button className="tx-link" onClick={() => setTable(!table)}><Icon name={table ? 'chart' : 'grid'} size={14} />{table ? 'Show chart' : 'Show as table'}</button>
    </div>
    {table ? <div className="table-scroll tx-table"><table><thead><tr><th>{xLabel.toUpperCase()}</th>{series.map(s => <th key={s.id}>{s.label.toUpperCase()}</th>)}</tr></thead>
      <tbody>{tableRows.map(at => <tr key={at}><td>{steps(at)}</td>{series.map(s => <td key={s.id}>{fmt(nearest(s.points, at)[1], 1)}</td>)}</tr>)}</tbody></table></div>
      : <div className="tx-chart" ref={box} tabIndex={0} role="img" aria-label={`${caption}. Use the left and right arrow keys to read values.`}
        onPointerMove={e => move(e.clientX)} onPointerLeave={() => setHoverX(null)} onKeyDown={key} onBlur={() => setHoverX(null)}>
        <svg viewBox={`0 0 ${W} ${H}`}>
          {ticks(yMin, yMax, 4).map(v => <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke={v === 0 ? INK.axis : INK.grid} strokeWidth={1} />
            <text x={L - 8} y={y(v) + 4} textAnchor="end" className="tx-tick">{fmt(v, 1)}</text></g>)}
          {ticks(xMin, xMax, 5).map(v => <text key={v} x={x(v)} y={H - B + 18} textAnchor="middle" className="tx-tick">{steps(v)}</text>)}
          <text x={L} y={H - 4} className="tx-axis-title">{xLabel}</text>
          <text x={12} y={T + 2} className="tx-axis-title" transform={`rotate(-90 12 ${T + 2})`} textAnchor="end">{yLabel}</text>
          {markers.filter(m => m.x >= xMin && m.x <= xMax).map(m => <g key={m.label}><line x1={x(m.x)} x2={x(m.x)} y1={T} y2={H - B} stroke={INK.axis} strokeWidth={1} />
            <text x={x(m.x) + 6} y={T + 12} className="tx-marker">{m.label}</text></g>)}
          {series.map(s => <polyline key={s.id} points={s.points.map(p => `${x(p[0])},${y(p[1])}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />)}
          {series.filter(s => s.dots).map(s => s.points.map(p => <circle key={`${s.id}-${p[0]}`} cx={x(p[0])} cy={y(p[1])} r={4} fill={s.color} stroke="#fffefb" strokeWidth={2} />))}
          {ends.map(({ s, last, y: ly }) => <g key={s.id}><line x1={x(last[0]) + 4} x2={W - R + 10} y1={y(last[1])} y2={ly} stroke={s.color} strokeWidth={1} opacity={0.6} />
            <text x={W - R + 14} y={ly - 2} className="tx-end-value">{fmt(last[1], 1)}</text><text x={W - R + 14} y={ly + 12} className="tx-end-label">{s.label}</text></g>)}
          {readout && <g><line x1={x(hoverX!)} x2={x(hoverX!)} y1={T} y2={H - B} stroke={INK.secondary} strokeWidth={1} />
            {readout.map(({ s, p }) => <circle key={s.id} cx={x(p[0])} cy={y(p[1])} r={5} fill={s.color} stroke="#fffefb" strokeWidth={2} />)}</g>}
        </svg>
        {readout && <div className="tx-tooltip" style={{ left: `${Math.min(68, (x(hoverX!) / W) * 100)}%` }}>
          <small>{steps(readout[0].p[0])} decisions</small>
          {readout.map(({ s, p }) => <div key={s.id}><i style={{ background: s.color }} /><b>{fmt(p[1], 1)}</b><span>{s.label}</span></div>)}
        </div>}
      </div>}
    <figcaption>{caption}</figcaption>
  </figure>
}

/* ---------- chapter 1: what it sees ----------------------------------------------- */
const PIPELINE = [
  ['Atari screen', '210 × 160 colour pixels, 60 times a second. Exactly what a human sees.'],
  ['Simplify', 'Turned grey and shrunk to 84 × 84. Colour and fine detail do not help it win, and smaller is faster.'],
  ['Stack 4 frames', 'The last four pictures are layered together, so movement becomes visible.'],
  ['Neural network', '3 layers that spot patterns (ball, paddles, gaps), then 512 “neurons” that combine them.'],
  ['One score per button', 'Out comes a Q-value for each button. The highest one is pressed.'],
]
function Court({ ball, faded = false, label }: { ball: [number, number]; faded?: boolean; label?: string }) {
  return <svg viewBox="0 0 84 84" className={faded ? 'tx-court faded' : 'tx-court'} role="img" aria-label={label}>
    <rect width="84" height="84" fill="#152019" /><rect x="4" y="22" width="3" height="16" fill="#e9b07c" /><rect x="77" y="40" width="3" height="16" fill="#b8e5a5" />
    <line x1="42" x2="42" y1="0" y2="84" stroke="#2c3b31" strokeDasharray="3 4" /><rect x={ball[0] - 2} y={ball[1] - 2} width="4" height="4" fill="#f4f4ec" />
  </svg>
}
function SeesChapter() {
  const [stage, setStage] = useState(2)
  const [four, setFour] = useState(true)
  const balls: [number, number][] = [[58, 58], [48, 50], [38, 42], [28, 34]]
  return <div className="tx-chapter-body">
    <ol className="tx-pipeline">{PIPELINE.map(([title], index) => <li key={title}>
      <button className={index === stage ? 'active' : ''} aria-pressed={index === stage} onClick={() => setStage(index)}><span>{index + 1}</span>{title}</button>
      {index < PIPELINE.length - 1 && <i aria-hidden="true">→</i>}</li>)}</ol>
    <p className="tx-explain"><b>{PIPELINE[stage][0]}.</b> {PIPELINE[stage][1]}</p>
    <div className="tx-frames-card">
      <div className="tx-toggle" role="group" aria-label="How many frames the agent sees">
        <button className={!four ? 'selected' : ''} aria-pressed={!four} onClick={() => setFour(false)}>1 frame</button>
        <button className={four ? 'selected' : ''} aria-pressed={four} onClick={() => setFour(true)}>4 frames</button>
      </div>
      <div className="tx-frames">{(four ? balls : balls.slice(-1)).map((ball, index, list) => <div key={index} className="tx-frame">
        <Court ball={ball} faded={four && index < list.length - 1} label={`Frame ${four ? index + 1 : 4}`} /><small>{four ? `t − ${list.length - 1 - index}` : 'now'}</small></div>)}
        {!four && <div className="tx-question"><b>?</b><span>Is the ball going up or down, fast or slow? From one picture you cannot tell.</span></div>}
        {four && <div className="tx-question ok"><Icon name="check" size={18} /><span>Layered together, the frames show the ball moving <b>up and to the left</b>, toward the orange paddle.</span></div>}
      </div>
    </div>
  </div>
}

/* ---------- chapter 2: how it decides --------------------------------------------- */
const SITUATIONS = [
  { id: 'high', title: 'Ball coming in high', ball: [28, 26] as [number, number], arrow: 'M34 40 L 12 22', q: { 'Move up': 0.62, Stay: 0.18, 'Move down': -0.35 } },
  { id: 'low', title: 'Ball coming in low', ball: [28, 60] as [number, number], arrow: 'M34 46 L 12 64', q: { 'Move up': -0.3, Stay: 0.15, 'Move down': 0.58 } },
  { id: 'away', title: 'Ball moving away', ball: [50, 42] as [number, number], arrow: 'M40 42 L 66 42', q: { 'Move up': 0.05, Stay: 0.12, 'Move down': 0.04 } },
]
function DecidesChapter() {
  const [pick, setPick] = useState(0)
  const [hover, setHover] = useState<string | null>(null)
  const situation = SITUATIONS[pick]
  const entries = Object.entries(situation.q)
  const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0]
  const zero = 50, scale = 42 // percent of the bar track
  return <div className="tx-chapter-body tx-two">
    <div>
      <div className="tx-chips" role="group" aria-label="Game situation">{SITUATIONS.map((item, index) => <button key={item.id} className={index === pick ? 'selected' : ''} aria-pressed={index === pick} onClick={() => setPick(index)}>{item.title}</button>)}</div>
      <svg viewBox="0 0 84 84" className="tx-court big" role="img" aria-label={situation.title}>
        <rect width="84" height="84" fill="#152019" /><rect x="4" y="34" width="3" height="16" fill="#e9b07c" /><rect x="77" y="36" width="3" height="16" fill="#b8e5a5" />
        <line x1="42" x2="42" y1="0" y2="84" stroke="#2c3b31" strokeDasharray="3 4" />
        <path d={situation.arrow} stroke="#f0a35e" strokeWidth="1.6" fill="none" markerEnd="url(#tx-head)" /><defs><marker id="tx-head" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 6 3 0 6Z" fill="#f0a35e" /></marker></defs>
        <rect x={situation.ball[0] - 2} y={situation.ball[1] - 2} width="4" height="4" fill="#f4f4ec" />
      </svg>
      <p className="tx-note">The DQN controls the orange paddle on the left, as in the two-player game. Values are illustrative, to show the idea.</p>
    </div>
    <div>
      <h4 className="tx-sub">The network’s <Term word="Q-value">Q-values</Term> for this screen</h4>
      <div className="tx-bars" role="list">{entries.map(([action, value]) => <div key={action} role="listitem" className={`tx-bar-row ${hover === action ? 'hover' : ''}`} tabIndex={0}
        onMouseEnter={() => setHover(action)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(action)} onBlur={() => setHover(null)}>
        <span className="tx-bar-label">{action}{action === best && <em>chosen</em>}</span>
        <span className="tx-bar-track"><span className="tx-bar-zero" style={{ left: `${zero}%` }} />
          <span className="tx-bar" style={{ left: `${value >= 0 ? zero : zero + value * scale}%`, width: `${Math.max(Math.abs(value) * scale, 0.8)}%`, background: action === best ? SERIES[0] : '#c9cfc6' }} /></span>
        <b className="tx-bar-value">{value > 0 ? '+' : ''}{fmt(value, 2)}</b>
        {hover === action && <span className="tx-bar-tip">{action === best ? 'Highest score, so this button is pressed.' : value < 0 ? 'Expected to lose points: avoid.' : 'Not the best option on this screen.'}</span>}
      </div>)}</div>
      <p className="tx-explain">A Q-value is the network’s prediction of <b>future points</b> if it presses that button now. It always presses the highest one, unless it is exploring (next chapter). The real Pong network scores 6 buttons; we show the 3 that matter.</p>
    </div>
  </div>
}

/* ---------- chapter 3: explore vs exploit ------------------------------------------ */
const epsilonAt = (decisions: number) => Math.max(0.1, 1 - 0.9 * decisions / 1_000_000)
function ExploreChapter() {
  const [progress, setProgress] = useState(250_000)
  const eps = epsilonAt(progress)
  const order = useMemo(() => Array.from({ length: 40 }, (_, i) => ((i * 23) % 40)), [])
  const randomCount = Math.round(eps * 40)
  const schedule: Point[] = Array.from({ length: 101 }, (_, i) => { const d = i * 100_000; return [d, epsilonAt(d)] })
  return <div className="tx-chapter-body">
    <p className="tx-explain">At first the network knows nothing, so its scores are noise. If it always trusted them it would repeat the same bad move forever. So it sometimes presses a <b>random</b> button to discover what happens. The chance of that is called <Term word="ε (epsilon)">ε</Term>.</p>
    <label className="tx-slider"><span>Training progress: <b>{steps(progress)}</b> of 10M decisions</span>
      <input type="range" min={0} max={10_000_000} step={50_000} value={progress} onChange={e => setProgress(Number(e.target.value))} aria-valuetext={`${steps(progress)} decisions, epsilon ${fmt(eps, 2)}`} /></label>
    <div className="tx-two">
      <div>
        <div className="tx-stat"><b>{fmt(eps * 100, 0)}%</b><span>of moves are random at this point</span></div>
        <div className="tx-dots" aria-label={`${randomCount} of the next 40 moves random`}>{order.map((slot, i) => <span key={i} className={slot < randomCount ? 'random' : 'chosen'} title={slot < randomCount ? 'Random move' : 'Best-known move'} />)}</div>
        <div className="tx-dot-key"><span><i className="random" />Random move (exploring)</span><span><i className="chosen" />Best Q-value (using what it knows)</span></div>
      </div>
      <div>
        <LineChart series={[{ id: 'eps', label: 'ε', color: SERIES[0], points: schedule }]} height={220} yLabel="ε (chance of random move)" yDomain={[0, 1.05]}
          markers={[{ x: progress, label: 'you are here' }]} caption="Exploration schedule used to pre-train the Pong and Breakout models: ε falls from 1.0 to 0.1 over the first million decisions, then stays at 0.1." />
      </div>
    </div>
    <p className="tx-note">In the two-player fine-tune ε went from 0.2 to 0.02. When you play it in the lab it uses ε 0.05: 1 move in 20 is random.</p>
  </div>
}

/* ---------- chapter 4: learning from mistakes --------------------------------------- */
const LOOP = [
  ['Play', 'Look at the screen, press the button with the highest Q-value (or a random one).', 'game'],
  ['Remember', 'Write the moment into replay memory: screen, button, reward, next screen.', 'download'],
  ['Practise', 'Pick 32 random moments from memory. Random so it does not just learn the last rally.', 'refresh'],
  ['Check the guess', 'Compare what it predicted with what actually happened: reward + the best score it now sees from the next screen.', 'check'],
  ['Adjust', 'Nudge millions of network weights a tiny bit so the guess moves toward reality. Then repeat.', 'brain'],
]
function LearnChapter() {
  const [step, setStep] = useState(0)
  const [auto, setAuto] = useState(false)
  const [gamma, setGamma] = useState(0.99)
  useEffect(() => { if (!auto) return; const id = setInterval(() => setStep(s => (s + 1) % LOOP.length), 1800); return () => clearInterval(id) }, [auto])
  const guess = 0.1, reward = 0, nextBest = 0.4, target = reward + gamma * nextBest, error = target - guess
  const future = Array.from({ length: 19 }, (_, i) => i * 5)
  return <div className="tx-chapter-body">
    <div className="tx-loop">
      <svg viewBox="0 0 320 320" className="tx-loop-ring" aria-hidden="true">
        <circle cx="160" cy="160" r="118" fill="none" stroke="#e3e9dd" strokeWidth="10" />
        <circle cx="160" cy="160" r="118" fill="none" stroke={SERIES[0]} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${(step + 1) / LOOP.length * 741} 741`} transform="rotate(-90 160 160)" />
      </svg>
      {LOOP.map(([title], index) => { const angle = -Math.PI / 2 + index / LOOP.length * Math.PI * 2; return <button key={title} className={`tx-loop-node ${index === step ? 'active' : ''}`} style={{ left: `${50 + Math.cos(angle) * 37}%`, top: `${50 + Math.sin(angle) * 37}%` }} aria-pressed={index === step} onClick={() => { setAuto(false); setStep(index) }}><Icon name={LOOP[index][2]} size={18} /><span>{title}</span></button> })}
      <div className="tx-loop-center"><span>STEP {step + 1} / {LOOP.length}</span><b>{LOOP[step][0]}</b><p>{LOOP[step][1]}</p>
        <div className="tx-loop-controls"><button className="button secondary" onClick={() => { setAuto(false); setStep((step + LOOP.length - 1) % LOOP.length) }} aria-label="Previous step">←</button><button className="button primary" onClick={() => setAuto(!auto)}><Icon name={auto ? 'stop' : 'play'} size={14} />{auto ? 'Pause' : 'Play the loop'}</button><button className="button secondary" onClick={() => { setAuto(false); setStep((step + 1) % LOOP.length) }} aria-label="Next step">→</button></div></div>
    </div>
    <div className="tx-two">
      <div className="tx-example">
        <h4 className="tx-sub">A worked example: one moment from memory</h4>
        <ol>
          <li><span>The network guessed pressing “up” was worth</span><b>{fmt(guess, 2)}</b></li>
          <li><span>The <Term word="reward" /> it actually got</span><b>{fmt(reward, 0)}</b></li>
          <li><span>Best Q-value on the next screen (from the <Term word="target network" />)</span><b>{fmt(nextBest, 2)}</b></li>
          <li><span>What it should have guessed: {fmt(reward, 0)} + {fmt(gamma, 3)} × {fmt(nextBest, 2)}</span><b>{fmt(target, 3)}</b></li>
          <li className="tx-total"><span>Error: it was {error > 0 ? 'too pessimistic' : 'too optimistic'}, so the weights are nudged to push this guess {error > 0 ? 'up' : 'down'}</span><b>{error > 0 ? '+' : ''}{fmt(error, 3)}</b></li>
        </ol>
        <p className="tx-note">This is repeated millions of times. Each nudge is tiny (learning rate 0.00025), so no single moment can wreck what it learned.</p>
      </div>
      <div>
        <h4 className="tx-sub">How much does a future point count today?</h4>
        <label className="tx-slider"><span>Patience γ (gamma): <b>{fmt(gamma, 3)}</b></span>
          <input type="range" min={0.9} max={0.999} step={0.001} value={gamma} onChange={e => setGamma(Number(e.target.value))} aria-valuetext={`gamma ${fmt(gamma, 3)}`} /></label>
        <div className="tx-columns" role="list" aria-label="Value today of a point won later">{future.map(n => { const v = gamma ** n; return <div key={n} role="listitem" className="tx-col" title={`${n} decisions ahead (${fmt(n / 15, 1)} s): worth ${fmt(v * 100, 0)}% today`}><span style={{ height: `${Math.max(v * 100, 1)}%` }} /></div> })}</div>
        <div className="tx-col-axis"><span>now</span><span>3 s</span><span>6 s ahead</span></div>
        <p className="tx-explain">With γ = {fmt(gamma, 3)}, a point won <b>1 second</b> from now is worth <b>{fmt(gamma ** 15 * 100, 0)}%</b> today and one <b>6 seconds</b> away <b>{fmt(gamma ** 90 * 100, 0)}%</b>. All our models used 0.99: patient enough to set up a rally, but still preferring points sooner.</p>
      </div>
    </div>
  </div>
}

/* ---------- chapter 5: the training data ------------------------------------------- */
type Transition = { step: number; frames: string; next_frame: string; action: string; action_label: string; reward: number; training_reward: number; game_over: boolean }
type Experience = {
  game: string; policy: string; seed: number; description: string; epsilon: number; decisions: number; score: number
  positive: number; negative: number; actions: { action: string; label: string; count: number }[]; transitions: Transition[]
}
type RunInfo = { id: string; label: string; kind: string }
type Records = { run: string; label: string; total: number; offset: number; columns: string[]; rows: string[][]; summary: { rows: number; games: number; decisions: number | null; hours: number | null } }

function ExperienceBrowser() {
  const [game, setGame] = useState<'pong' | 'breakout'>('breakout')
  const [policy, setPolicy] = useState<'random' | 'trained'>('trained')
  const [seed, setSeed] = useState(7)
  const [data, setData] = useState<Experience | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  useEffect(() => {
    let alive = true
    setLoading(true); setError(null)
    api<Experience>(`/api/training/experience?game=${game}&policy=${policy}&seed=${seed}`)
      .then(value => { if (alive) { setData(value); setOpen(null) } })
      .catch(err => { if (alive) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [game, policy, seed])
  const zero = data ? data.decisions - data.positive - data.negative : 0
  const maxCount = data ? Math.max(...data.actions.map(a => a.count), 1) : 1
  const selected = data && open !== null ? data.transitions[open] : null
  return <div className="tx-data">
    <div className="tx-data-controls">
      <div className="tx-toggle" role="group" aria-label="Game">{(['pong', 'breakout'] as const).map(id => <button key={id} className={game === id ? 'selected' : ''} aria-pressed={game === id} onClick={() => setGame(id)}>{id === 'pong' ? 'Pong' : 'Breakout'}</button>)}</div>
      <div className="tx-toggle" role="group" aria-label="Who is playing">{(['random', 'trained'] as const).map(id => <button key={id} className={policy === id ? 'selected' : ''} aria-pressed={policy === id} onClick={() => setPolicy(id)}>{id === 'random' ? 'Start of training (random)' : 'Trained model'}</button>)}</div>
      <button className="tx-link" disabled={loading} onClick={() => setSeed(s => (s + 1) % 10000)}><Icon name="refresh" size={14} />Record a different stretch</button>
    </div>
    {error && <p className="tx-note">Could not record a sample: {error}</p>}
    {!data && loading && <div className="tx-chart-empty">Playing {SAMPLE_SECONDS} seconds of real Atari to record experience…</div>}
    {data && <div className={loading ? 'tx-data-body loading' : 'tx-data-body'}>
      <p className="tx-note">{data.description} Recorded live just now: {fmt(data.decisions, 0)} decisions (about {fmt(data.decisions / 15, 0)} seconds of play) at ε {fmt(data.epsilon, 2)}{data.policy === 'random' ? ', which is how every model starts' : ''}.</p>
      <div className="tx-data-stats">
        <div><b>{fmt(data.decisions, 0)}</b><span>experiences recorded</span></div>
        <div><b>{fmt(zero / Math.max(data.decisions, 1) * 100, 1)}%</b><span>had reward 0: nothing happened yet</span></div>
        <div><b className="pos">+{fmt(data.positive, 0)}</b><span>moments rewarded +1</span></div>
        <div><b className="neg">{data.negative ? `−${fmt(data.negative, 0)}` : '0'}</b><span>{data.game === 'breakout' ? 'moments penalised: Breakout never subtracts points (a lost ball just costs a life)' : 'moments penalised −1 (a point lost)'}</span></div>
      </div>
      <div className="tx-two">
        <div>
          <h4 className="tx-sub">Which buttons it pressed</h4>
          <div className="tx-bars compact" role="list">{data.actions.map(item => <div key={item.action} role="listitem" className="tx-bar-row" title={`${item.label}: ${item.count} of ${data.decisions} decisions`}>
            <span className="tx-bar-label">{item.label}</span>
            <span className="tx-bar-track"><span className="tx-bar" style={{ left: 0, width: `${Math.max(item.count / maxCount * 100, 0.8)}%`, background: SERIES[0] }} /></span>
            <b className="tx-bar-value">{fmt(item.count / Math.max(data.decisions, 1) * 100, 0)}%</b></div>)}</div>
          <p className="tx-note">{data.policy === 'random' ? 'Random play presses every button about equally often.' : 'The trained model has favourites: it learned which buttons pay off.'}</p>
        </div>
        <div>
          <h4 className="tx-sub">Why learning takes millions of these</h4>
          <p className="tx-explain">Only <b>{fmt((data.positive + data.negative) / Math.max(data.decisions, 1) * 100, 1)}%</b> of these moments carry any reward. Everything else must be learned indirectly, by linking quiet moments to points that come later. Each model learned from about <b>10 million</b> such experiences; its <Term word="replay memory" /> held the most recent 200k (Breakout) to 1M (Pong) at a time.</p>
        </div>
      </div>
      <h4 className="tx-sub">Individual experiences: exactly what the network trains on</h4>
      <div className="tx-transitions">{data.transitions.map((item, index) => <button key={item.step} className={`tx-transition ${open === index ? 'open' : ''} ${item.reward > 0 ? 'pos' : item.reward < 0 ? 'neg' : ''}`} onClick={() => setOpen(open === index ? null : index)} aria-expanded={open === index}>
        <span className="tx-t-head"><span>Moment #{item.step}</span><span className={`tx-reward ${item.reward > 0 ? 'pos' : item.reward < 0 ? 'neg' : ''}`}>{item.reward > 0 ? '+' : item.reward < 0 ? '−' : ''}{fmt(Math.abs(item.training_reward), 0)}</span></span>
        <img src={item.frames} alt={`Four stacked frames before moment ${item.step}`} className="tx-strip" />
        <span className="tx-t-foot"><span className="tx-pill">{item.action_label}</span><span>{item.game_over ? 'game over' : item.reward !== 0 ? (item.reward > 0 ? 'point won' : 'point lost') : 'no reward'}</span></span>
      </button>)}</div>
      {selected && <div className="tx-detail" role="region" aria-label={`Experience ${selected.step} in detail`}>
        <div className="tx-detail-part"><span>1 · STATE: what it saw (4 frames, oldest first)</span><img src={selected.frames} alt="State: four stacked frames" className="tx-strip big" /></div>
        <div className="tx-detail-row">
          <div className="tx-detail-part"><span>2 · ACTION</span><b>{selected.action_label}</b><small>{selected.action}</small></div>
          <div className="tx-detail-part"><span>3 · REWARD</span><b className={selected.reward > 0 ? 'pos' : selected.reward < 0 ? 'neg' : ''}>{selected.reward > 0 ? '+' : ''}{fmt(selected.reward, 0)}</b><small>{selected.reward !== 0 ? `stored as ${selected.training_reward > 0 ? '+' : '−'}1` : 'nothing happened yet'}</small></div>
          <div className="tx-detail-part"><span>4 · NEXT FRAME</span><img src={selected.next_frame} alt="Next frame after the action" className="tx-next" /></div>
          <div className="tx-detail-part"><span>5 · GAME OVER?</span><b>{selected.game_over ? 'Yes' : 'No'}</b><small>{selected.game_over ? 'no future points to add' : 'learning adds the best future score'}</small></div>
        </div>
        <p className="tx-note">These five fields are one row of training data. The next frame is added to the stack to form the next state.</p>
      </div>}
      <p className="tx-note">Click a moment to see its five parts. Images are the real 84 × 84 greyscale frames the network receives.</p>
    </div>}
  </div>
}

function RecordsBrowser() {
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [run, setRun] = useState('pong_nature')
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<Records | null>(null)
  const [error, setError] = useState<string | null>(null)
  const limit = 25
  useEffect(() => { api<RunInfo[]>('/api/training/runs').then(setRuns).catch(() => { /* list is optional */ }) }, [])
  useEffect(() => {
    let alive = true
    api<Records>(`/api/training/records?run=${run}&offset=${offset}&limit=${limit}`).then(value => { if (alive) { setData(value); setError(null) } })
      .catch(err => { if (alive) setError(err instanceof Error ? err.message : String(err)) })
    return () => { alive = false }
  }, [run, offset])
  const total = data?.total ?? 0
  const go = (value: number) => setOffset(Math.max(0, Math.min(Math.max(total - limit, 0), value)))
  const kinds: Record<string, string> = { episode: 'Game', game: 'Game', validation: 'Test', eval: 'Test', evaluation: 'Test' }
  return <div className="tx-records">
    <div className="tx-data-controls">
      <label className="tx-select">Training run<select value={run} onChange={e => { setRun(e.target.value); setOffset(0) }}>{runs.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
      <a className="button secondary" href={`/api/training/records/${run}.csv`} download><Icon name="download" size={14} />Download full CSV</a>
    </div>
    {error && <p className="tx-note">Could not load this log: {error}</p>}
    {data && <>
      <div className="tx-data-stats small">
        <div><b>{fmt(data.summary.games, 0)}</b><span>games played</span></div>
        <div><b>{data.summary.decisions ? steps(data.summary.decisions) : '—'}</b><span>decisions</span></div>
        <div><b>{fmt(data.summary.hours, 1)} h</b><span>on the GPU</span></div>
        <div><b>{fmt(data.summary.rows, 0)}</b><span>rows in the log</span></div>
      </div>
      <div className="table-scroll tx-table tx-log"><table><thead><tr>{data.columns.map(column => <th key={column}>{column === 'ε' ? column : column.toUpperCase()}</th>)}</tr></thead>
        <tbody>{data.rows.map((row, index) => <tr key={offset + index} className={kinds[row[0]] === 'Test' ? 'test-row' : ''}>{row.map((cell, i) => <td key={i}>{i === 0 ? (kinds[cell] ?? cell) : cell || '—'}</td>)}</tr>)}</tbody></table></div>
      <div className="tx-pager">
        <span>Rows {fmt(offset + 1, 0)}–{fmt(Math.min(offset + limit, total), 0)} of {fmt(total, 0)}</span>
        <div><button className="button secondary" disabled={offset === 0} onClick={() => go(0)}>Start</button><button className="button secondary" disabled={offset === 0} onClick={() => go(offset - limit)}>← Earlier</button>
          <button className="button secondary" onClick={() => go(Math.floor(total / 2))}>Middle</button>
          <button className="button secondary" disabled={offset + limit >= total} onClick={() => go(offset + limit)}>Later →</button><button className="button secondary" disabled={offset + limit >= total} onClick={() => go(total - limit)}>End</button></div>
      </div>
      <p className="tx-note">One row per game. “Test” rows (highlighted) are check-ups with exploration almost off. Compare the start, where scores sit near the minimum and ε is high, with the end. Loss is blank before learning starts.</p>
    </>}
  </div>
}

const SAMPLE_SECONDS = 40
function DataChapter() {
  const [tab, setTab] = useState<'experience' | 'logs'>('experience')
  return <div className="tx-chapter-body">
    <p className="tx-explain">There is no ready-made dataset of good moves. A DQN <b>creates its own training data</b> by playing: every <Term word="decision" /> becomes one experience of five parts. Below, the lab records real experience right now, and you can browse the logs written during the actual training runs.</p>
    <div className="tx-tabs" role="tablist">{([['experience', 'Experience it learns from'], ['logs', 'Training logs']] as const).map(([id, title]) => <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'selected' : ''} onClick={() => setTab(id)}>{title}</button>)}</div>
    {tab === 'experience' ? <ExperienceBrowser /> : <RecordsBrowser />}
  </div>
}

/* ---------- chapter 6: the real training runs ------------------------------------- */
function RunsChapter() {
  const [curves, setCurves] = useState<Curves | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'pretrain' | 'finetune' | 'breakout'>('pretrain')
  useEffect(() => { api<Curves>('/api/training/curves').then(setCurves).catch(err => setError(err instanceof Error ? err.message : String(err))) }, [])
  const tabs = [['pretrain', 'Pong · pre-training'], ['finetune', 'Pong · two-player fine-tune'], ['breakout', 'Breakout']] as const
  const view = !curves ? null : tab === 'pretrain' ? {
    series: [{ id: 'nature', label: 'Nature DQN', color: SERIES[0], points: curves.pong_single.training.nature }, { id: 'dueling', label: 'Dueling DQN', color: SERIES[1], points: curves.pong_single.training.dueling }],
    yLabel: 'Points per game (21 = perfect)', yDomain: [-21, 21] as [number, number],
    markers: [{ x: 1_000_000, label: 'exploration settles (ε 0.1)' }],
    caption: `Average score over the last ${curves.window} games while training against Atari’s built-in opponent. Scores start near −20 (losing almost every point) and end above +12. Exploration (ε 0.1) is still on here, so the separate test scores are higher.`,
    story: [['Decisions 0–1M', 'Mostly random play. It loses nearly every point (about −19 to −20).'], ['1M–4M', 'It learns to return the ball. The average climbs to about zero by 4M decisions (Nature first, Dueling a little later).'], ['4M–10M', 'It wins more and more points: the training average ends at +15.8 (Nature) and +12.1 (Dueling); test scores +19.1 and +17.2.']],
  } : tab === 'finetune' ? {
    series: [{ id: 'nature', label: 'Nature DQN', color: SERIES[0], points: curves.pong_two_player.evaluation.nature, dots: true }, { id: 'dueling', label: 'Dueling DQN', color: SERIES[1], points: curves.pong_two_player.evaluation.dueling, dots: true }],
    yLabel: 'Test score (points won − lost)', yDomain: [-21, 21] as [number, number], markers: [],
    caption: 'Test score every 100k decisions: 8 games, exploration off, against the hardest scripted opponent. Both models started from their pre-trained weights, mirrored to the other paddle.',
    story: [['First 1.2M decisions', 'Test scores swing between about −20 and +12 while it adapts to the other paddle and a new kind of opponent.'], ['The −55 dip at 1.5M', 'Pong also gives −1 each time a player fails to serve in time. For one test the Nature model stopped serving and was penalised again and again, so it lost by more than 21. It recovered by the next test.'], ['Around 2M', 'Both pass +15 for the first time (Nature at 1.8M, Dueling at 2.0M).'], ['Best', 'Nature reached +21.0 (a shut-out) at 2.9M; Dueling +19.0 at 3M. These are the models you play.']],
  } : {
    series: [{ id: 'uniform', label: 'Uniform replay', color: SERIES[0], points: curves.breakout.training.uniform }, { id: 'prioritized', label: 'Prioritized replay', color: SERIES[1], points: curves.breakout.training.prioritized }, { id: 'upstream', label: 'Upstream setup', color: SERIES[2], points: curves.breakout.training.upstream }],
    yLabel: 'Bricks score per episode', yDomain: undefined, markers: [{ x: 1_000_000, label: 'exploration settles' }],
    caption: `Average score over the last ${curves.window} episodes of Breakout training, exploration on (ε about 0.1). The upstream run counts every lost life as an episode, so its line is per life, not per game.`,
    story: [['Uniform', 'Random pages from memory. Best here: 46.6 average, best single game 257.'], ['Prioritized', 'Re-reads surprising moments more often. Ahead early (10.4 vs 5.6 at 2M), overtaken around 6M, ended at 29.9.'], ['Upstream', 'Settings from the reference code (smaller memory, Adam). 12.9 per life.']],
  }
  return <div className="tx-chapter-body">
    <p className="tx-explain">These are the real records of the models in this lab, trained on a GPU in November 2026. Each line is an average over the last {curves?.window ?? 100} games, because single games are noisy. Hover or use the arrow keys to read exact values.</p>
    <div className="tx-tabs" role="tablist">{tabs.map(([id, title]) => <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'selected' : ''} onClick={() => setTab(id)}>{title}</button>)}</div>
    {error && <p className="tx-note">Training curves could not be loaded: {error}</p>}
    {view && <div className="tx-runs">
      <LineChart series={view.series} yLabel={view.yLabel} yDomain={view.yDomain} markers={view.markers} caption={view.caption} />
      <ol className="tx-story">{view.story.map(([title, text]) => <li key={title}><b>{title}</b><p>{text}</p></li>)}</ol>
    </div>}
  </div>
}

/* ---------- chapter 7: the models --------------------------------------------------- */
const MODELS = [
  { game: 'Pong', name: 'Nature DQN', score: '+21.0', unit: 'best test vs hard opponent', time: '4.4 h + 2.3 h', note: 'Pre-trained 10M decisions, fine-tuned 3M in two-player Pong. The 2015 paper’s network.', view: 'pong' as const },
  { game: 'Pong', name: 'Dueling DQN', score: '+19.0', unit: 'best test vs hard opponent', time: '4.8 h + 2.3 h', note: 'Same training; the network separately estimates “how good is this moment” and “how much better is each button”.', view: 'pong' as const },
  { game: 'Breakout', name: 'Uniform replay', score: '46.6', unit: 'training average, last 100 games', time: '6.3 h', note: '10M decisions, 200k-moment memory sampled evenly.', view: 'breakout' as const },
  { game: 'Breakout', name: 'Prioritized replay', score: '29.9', unit: 'training average, last 100 games', time: '7.2 h', note: '10M decisions; surprising moments are replayed more often.', view: 'breakout' as const },
  { game: 'Breakout', name: 'Upstream setup', score: '12.9', unit: 'training average per life', time: '6.6 h', note: '10M decisions with the reference code’s settings (Adam, 10k memory).', view: 'breakout' as const },
]
function ModelsChapter({ onPlay }: { onPlay: (view: 'pong' | 'breakout') => void }) {
  return <div className="tx-chapter-body">
    <div className="tx-model-grid">{MODELS.map(model => <article key={model.name + model.game} className="tx-model">
      <span className={`tx-game ${model.game.toLowerCase()}`}>{model.game}</span><h4>{model.name}</h4>
      <div className="tx-model-score"><b>{model.score}</b><small>{model.unit}</small></div>
      <p>{model.note}</p><div className="tx-model-foot"><span><Icon name="history" size={13} />{model.time} on GPU</span><button className="tx-link" onClick={() => onPlay(model.view)}>Play it<Icon name="arrow" size={13} /></button></div>
    </article>)}</div>
    <p className="tx-note">Files, settings and full per-game records are in the repository’s <code>trained_models/</code> folder.</p>
  </div>
}

/* ---------- the page ------------------------------------------------------------- */
const CHAPTERS = [
  { id: 'sees', title: 'What it sees', blurb: 'Pixels in, just like you.' },
  { id: 'decides', title: 'How it decides', blurb: 'A score for every button.' },
  { id: 'explore', title: 'Explore or exploit', blurb: 'Why it sometimes plays randomly.' },
  { id: 'learn', title: 'Learning from mistakes', blurb: 'Guess, check, adjust, repeat.' },
  { id: 'data', title: 'The training data', blurb: 'Experience it collects itself.' },
  { id: 'runs', title: 'Our training runs', blurb: 'Real curves from the GPU.' },
  { id: 'models', title: 'Meet the models', blurb: 'Pick one and play it.' },
]
export function TrainingExplained({ onPlay }: { onPlay: (view: 'pong' | 'breakout') => void }) {
  const [active, setActive] = useState('sees')
  useEffect(() => {
    const observer = new IntersectionObserver(entries => { const seen = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]; if (seen) setActive(seen.target.id.replace('tx-', '')) }, { rootMargin: '-20% 0px -65% 0px' })
    CHAPTERS.forEach(chapter => { const node = document.getElementById(`tx-${chapter.id}`); if (node) observer.observe(node) })
    return () => observer.disconnect()
  }, [])
  const body: Record<string, React.ReactNode> = { sees: <SeesChapter />, decides: <DecidesChapter />, explore: <ExploreChapter />, learn: <LearnChapter />, data: <DataChapter />, runs: <RunsChapter />, models: <ModelsChapter onPlay={onPlay} /> }
  return <div className="tx-page">
    <section className="tx-hero">
      <div><span className="section-index">NO MATHS NEEDED · ABOUT 5 MINUTES</span><h2>How does a computer learn to play Atari on its own?</h2>
        <p>Nobody showed these models how to play. Each one started out pressing random buttons and improved only from the score: +1 for a point won, −1 for a point lost. This technique is called <b>Deep Q-Learning</b>. Click through the seven steps below; underlined words like <Term word="decision" /> or <Term word="replay memory" /> explain themselves.</p></div>
      <div className="tx-hero-stats"><div><b>0</b><span>human examples used</span></div><div><b>10M</b><span>decisions per model: almost 8 days of non-stop play at game speed</span></div><div><b>+21</b><span>best Pong test score (21 = every point)</span></div></div>
    </section>
    <nav className="tx-nav" aria-label="Chapters">{CHAPTERS.map((chapter, index) => <a key={chapter.id} href={`#tx-${chapter.id}`} className={active === chapter.id ? 'active' : ''} onClick={event => { event.preventDefault(); document.getElementById(`tx-${chapter.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}><span>{String(index + 1).padStart(2, '0')}</span>{chapter.title}</a>)}</nav>
    {CHAPTERS.map((chapter, index) => <section key={chapter.id} id={`tx-${chapter.id}`} className="history-card tx-chapter" aria-labelledby={`tx-${chapter.id}-title`}>
      <div className="card-heading"><div><span className="section-index">{String(index + 1).padStart(2, '0')} / {chapter.blurb.toUpperCase()}</span><h2 id={`tx-${chapter.id}-title`}>{chapter.title}</h2></div></div>
      {body[chapter.id]}
    </section>)}
    <section className="history-card tx-chapter" aria-labelledby="tx-glossary-title"><div className="card-heading"><div><span className="section-index">QUICK REFERENCE</span><h2 id="tx-glossary-title">Glossary</h2></div></div>
      <dl className="tx-glossary">{Object.entries(GLOSSARY).map(([word, text]) => <div key={word}><dt>{word}</dt><dd>{text}</dd></div>)}</dl></section>
  </div>
}
