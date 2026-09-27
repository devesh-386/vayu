import type { ModelReport } from '../lib/api'
import { Reveal } from './Reveal'

const pct = (v: number) => `${Math.round(v * 100)}%`
const BASELINE = 'Persistence (baseline)'
const COLORS: Record<string, string> = {
  [BASELINE]: 'var(--faint)',
  'Linear Regression': '#9aa3b2',
  'Random Forest': '#6f86c9',
  XGBoost: '#4b6bd6',
  LSTM: 'var(--accent)',
}

export function ModelProof({ report }: { report: ModelReport | null }) {
  return (
    <section id="model" className="border-y border-line bg-surface">
      <div className="mx-auto max-w-[1320px] px-5 py-24 md:px-8 md:py-32">
        <Reveal>
          <h2 className="max-w-[18ch] text-3xl font-semibold tracking-[-0.025em] md:text-5xl">
            Scored on fourteen months it never saw.
          </h2>
          <p className="mt-4 max-w-[62ch] text-muted">
            Every model is judged against the simplest honest guess: tomorrow's air will be the same as today's.
          </p>
        </Reveal>
        {!report ? <div className="skeleton mt-12 h-72" /> : <Body r={report} />}
      </div>
    </section>
  )
}

function Body({ r }: { r: ModelReport }) {
  const hs = Object.keys(r.horizons).sort()
  const h1 = r.horizons[hs[0]], h3 = r.horizons[hs[hs.length - 1]]
  const best1 = h1.best_model
  const gain1 = 1 - h1.test_mae[best1] / h1.test_mae[BASELINE]
  const best3 = Object.entries(h3.test_mae).filter(([n]) => n !== BASELINE).sort((a, b) => a[1] - b[1])[0]
  const gain3 = 1 - best3[1] / h3.test_mae[BASELINE]
  const recall = Math.max(...Object.entries(r.results).filter(([n]) => n !== BASELINE).map(([, m]) => m.alert_recall))

  const stats = [
    { value: pct(gain1), label: `smaller error than "same as today" for tomorrow (${best1})` },
    { value: pct(gain3), label: `smaller error three days out (${best3[0]})` },
    { value: pct(recall), label: 'of Poor or worse days flagged a day in advance' },
  ]

  return (
    <>
      <div className="mt-14 grid gap-10 md:grid-cols-3">
        {stats.map((s, i) => (
          <Reveal key={s.label} delay={i * 0.08}>
            <p className="tabular font-mono text-6xl font-medium tracking-[-0.04em] text-accent md:text-7xl">{s.value}</p>
            <p className="mt-3 max-w-[28ch] text-muted">{s.label}</p>
          </Reveal>
        ))}
      </div>

      <div className="mt-20 grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Reveal>
          <h3 className="font-medium">Average error by days ahead</h3>
          <p className="mt-1 text-sm text-muted">Mean absolute error in AQI points. Lower is better.</p>
          <HorizonChart r={r} hs={hs} />
        </Reveal>
        <Reveal delay={0.08}>
          <h3 className="font-medium">Next-day results</h3>
          <p className="mt-1 text-sm text-muted">
            Test set {r.split.test[0]} to {r.split.test[1]}, {r.split.test[2].toLocaleString('en-IN')} city-days.
          </p>
          <table className="mt-6 w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr><th className="pb-3 font-normal">Model</th><th className="pb-3 text-right font-normal">Error</th>
                <th className="pb-3 text-right font-normal">R²</th><th className="pb-3 text-right font-normal">Right band</th></tr>
            </thead>
            <tbody>
              {Object.entries(r.results).map(([n, m]) => (
                <tr key={n} className={n === best1 ? 'font-medium' : 'text-muted'}>
                  <td className="py-2.5">
                    <span className="mr-2 inline-block h-2 w-2 rounded-full align-middle" style={{ background: COLORS[n] }} aria-hidden />
                    {n === BASELINE ? 'Same as today' : n}
                  </td>
                  <td className="tabular py-2.5 text-right font-mono">{m.MAE.toFixed(1)}</td>
                  <td className="tabular py-2.5 text-right font-mono">{m.R2.toFixed(2)}</td>
                  <td className="tabular py-2.5 text-right font-mono">{pct(m.category_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-xs text-muted">
            80% forecast ranges contained the actual AQI on {pct(h1.interval.coverage_80)} of test days.
          </p>
        </Reveal>
      </div>
    </>
  )
}

function HorizonChart({ r, hs }: { r: ModelReport; hs: string[] }) {
  const W = 560, H = 280, pad = { l: 36, r: 110, t: 16, b: 32 }
  const models = Object.keys(r.horizons[hs[0]].test_mae)
  const all = hs.flatMap(h => Object.values(r.horizons[h].test_mae))
  const lo = Math.floor(Math.min(...all) / 5) * 5, hi = Math.ceil(Math.max(...all) / 5) * 5
  const x = (i: number) => pad.l + (i / (hs.length - 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b)
  const ticks = Array.from({ length: (hi - lo) / 5 + 1 }, (_, i) => lo + i * 5)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-6 w-full" role="img" aria-label="Error by forecast horizon for each model">
      {ticks.map(t => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" className="fill-[var(--faint)] font-mono text-[10px]">{t}</text>
        </g>
      ))}
      {hs.map((h, i) => (
        <text key={h} x={x(i)} y={H - 8} textAnchor="middle" className="fill-[var(--faint)] text-[11px]">
          {h === '1' ? '1 day' : `${h} days`}
        </text>
      ))}
      {models.map(m => {
        const vals = hs.map(h => r.horizons[h].test_mae[m])
        return (
          <g key={m}>
            <polyline points={vals.map((v, i) => `${x(i)},${y(v)}`).join(' ')} fill="none" stroke={COLORS[m]}
              strokeWidth={m === 'LSTM' || m === 'XGBoost' ? 2.25 : 1.5} strokeDasharray={m === BASELINE ? '5 4' : undefined} />
            {vals.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={3} fill={COLORS[m]} />)}
          </g>
        )
      })}
      {endLabels(models.map(m => ({ m, y: y(r.horizons[hs[hs.length - 1]].test_mae[m]) }))).map(({ m, y: ly }) => (
        <text key={m} x={x(hs.length - 1) + 10} y={ly + 4} className="fill-[var(--muted)] text-[11px]">
          {m === BASELINE ? 'Same as today' : m}
        </text>
      ))}
    </svg>
  )
}

/** Pushes line-end labels apart so close values do not overprint each other. */
function endLabels(items: { m: string; y: number }[], gap = 13) {
  const sorted = [...items].sort((a, b) => a.y - b.y)
  for (let i = 1; i < sorted.length; i++) sorted[i].y = Math.max(sorted[i].y, sorted[i - 1].y + gap)
  return sorted
}
