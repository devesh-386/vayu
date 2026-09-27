import { useLayoutEffect, useRef, useState } from 'react'
import type { CityData } from '../lib/api'
import { BANDS, band, colorFor, shortDate } from '../lib/aqi'

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(720)
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

type Pt = { date: string; aqi: number; low?: number; high?: number; forecast: boolean }

export function ForecastChart({ data }: { data: CityData }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const height = 300
  const pad = { l: 36, r: 12, t: 16, b: 28 }

  const pts: Pt[] = [
    ...data.history.map(h => ({ ...h, forecast: false })),
    ...data.outlook.map(o => ({ date: o.date, aqi: o.aqi, low: o.low, high: o.high, forecast: true })),
  ]
  const n = pts.length
  const ymax = Math.max(120, ...pts.map(p => p.high ?? p.aqi)) * 1.1
  const x = (i: number) => pad.l + (i / (n - 1)) * (width - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - v / ymax) * (height - pad.t - pad.b)
  const lastObs = data.history.length - 1

  const line = (from: number, to: number) =>
    pts.slice(from, to + 1).map((p, k) => `${k ? 'L' : 'M'}${x(from + k).toFixed(1)},${y(p.aqi).toFixed(1)}`).join('')
  const rangeIdx = [lastObs, ...pts.map((p, i) => (p.forecast ? i : -1)).filter(i => i >= 0)]
  const range =
    rangeIdx.map((i, k) => `${k ? 'L' : 'M'}${x(i)},${y(pts[i].high ?? pts[i].aqi)}`).join('') +
    [...rangeIdx].reverse().map(i => `L${x(i)},${y(pts[i].low ?? pts[i].aqi)}`).join('') + 'Z'

  const ticks = BANDS.map(b => b.max).filter(v => v < ymax)
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const i = Math.round(((e.clientX - r.left - pad.l) / (width - pad.l - pad.r)) * (n - 1))
    setHover(Math.max(0, Math.min(n - 1, i)))
  }
  const h = hover != null ? pts[hover] : null

  return (
    <div ref={ref} className="relative w-full">
      <svg width={width} height={height} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        aria-label={`AQI in ${data.city}: last 30 days and a three-day forecast`} className="block touch-none">
        {BANDS.map((b, i) => {
          const lo = i ? BANDS[i - 1].max : 0
          if (lo >= ymax) return null
          return <rect key={b.name} x={pad.l} width={width - pad.l - pad.r} y={y(Math.min(b.max, ymax))}
            height={y(lo) - y(Math.min(b.max, ymax))} fill={b.color} opacity={0.07} />
        })}
        {ticks.map(t => (
          <g key={t}>
            <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" className="fill-[var(--faint)] font-mono text-[10px]">{t}</text>
          </g>
        ))}
        <rect x={x(lastObs)} width={width - pad.r - x(lastObs)} y={pad.t} height={height - pad.t - pad.b}
          fill="var(--surface-2)" opacity={0.6} />
        <path d={range} fill="var(--accent)" opacity={0.14} />
        <path d={line(0, lastObs)} fill="none" stroke="var(--ink)" strokeWidth={1.75} strokeLinejoin="round" />
        <path d={line(lastObs, n - 1)} fill="none" stroke="var(--accent)" strokeWidth={2} strokeDasharray="5 4" />
        {pts.map((p, i) => (p.forecast ? (
          <circle key={i} cx={x(i)} cy={y(p.aqi)} r={5} fill={colorFor(p.aqi)} stroke="var(--surface)" strokeWidth={2} />
        ) : null))}
        {[0, Math.floor(lastObs / 2), lastObs, ...(x(n - 1) - x(lastObs) > 70 ? [n - 1] : [])].map(i => (
          <text key={i} x={x(i)} y={height - 8} textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}
            className="fill-[var(--faint)] text-[11px]">{i === lastObs ? 'Today' : shortDate(pts[i].date)}</text>
        ))}
        {h && hover != null && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={height - pad.b} stroke="var(--faint)" strokeDasharray="2 3" />
            <circle cx={x(hover)} cy={y(h.aqi)} r={4} fill="var(--ink)" />
          </g>
        )}
      </svg>
      {h && hover != null && (
        <div className="pointer-events-none absolute top-2 rounded-xl border border-line bg-surface px-3 py-2 text-sm shadow-lg"
          style={{ left: Math.min(Math.max(x(hover) - 70, 0), width - 150) }}>
          <p className="text-xs text-muted">{shortDate(h.date)}{h.forecast ? ', forecast' : ''}</p>
          <p className="tabular font-mono text-lg font-medium">{h.aqi} <span className="font-sans text-sm font-normal text-muted">{band(h.aqi).name}</span></p>
          {h.forecast && <p className="text-xs text-muted">80% range {h.low} to {h.high}</p>}
        </div>
      )}
    </div>
  )
}
