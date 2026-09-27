import { ArrowsClockwiseIcon } from '@phosphor-icons/react'
import type { OverviewRow } from '../lib/api'
import { colorFor, dayLabel, inkOn } from '../lib/aqi'
import { Reveal } from './Reveal'

type Props = {
  rows: OverviewRow[] | null
  loading: boolean
  error: string | null
  selected: string
  onSelect: (c: string) => void
  onRetry: () => void
}

export function CityBoard({ rows, loading, error, selected, onSelect, onRetry }: Props) {
  const firstOk = rows?.find(r => 'today' in r)
  const days = firstOk && 'today' in firstOk ? [firstOk.today.date, ...firstOk.outlook.map(o => o.date)] : []

  return (
    <section id="cities" className="mx-auto max-w-[1320px] px-5 py-24 md:px-8 md:py-32">
      <Reveal>
        <h2 className="text-3xl font-semibold tracking-[-0.025em] md:text-5xl">Every city, four days.</h2>
        <p className="mt-4 max-w-[60ch] text-muted">
          Today's measured AQI and the model's forecast for the next three days. Pick a city to see why.
        </p>
      </Reveal>

      <div className="mt-12">
        {error ? (
          <div className="panel flex flex-wrap items-center justify-between gap-4 p-6">
            <p className="text-muted">Could not load the city board: {error}</p>
            <button onClick={onRetry} className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm hover:bg-surface-2">
              <ArrowsClockwiseIcon size={16} /> Try again
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto overflow-y-hidden py-1 [scrollbar-width:thin]">
            <div role="table" aria-label="AQI by city" className="min-w-[560px]">
              <div role="row" className="grid grid-cols-[minmax(120px,1.4fr)_repeat(4,minmax(88px,1fr))] gap-2 px-3 pb-3 text-xs text-muted">
                <span role="columnheader">City</span>
                {(days.length ? days : ['', '', '', '']).map((d, i) => (
                  <span role="columnheader" key={i}>{d ? dayLabel(d, days[0]) : ''}</span>
                ))}
              </div>
              <div className="space-y-2">
                {loading || !rows
                  ? Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton h-16" />)
                  : rows.map((r, idx) => {
                      if (!('today' in r)) {
                        return (
                          <div role="row" key={r.city} className="grid grid-cols-[minmax(120px,1.4fr)_1fr] items-center gap-2 rounded-2xl px-3 py-4">
                            <span className="font-medium">{r.city}</span>
                            <span className="text-sm text-muted">No data right now</span>
                          </div>
                        )
                      }
                      const cells = [r.today.aqi, ...r.outlook.map(o => o.aqi)]
                      const active = r.city === selected
                      return (
                        <Reveal key={r.city} delay={idx * 0.04}>
                          <button role="row" onClick={() => { onSelect(r.city); document.getElementById('forecast')?.scrollIntoView() }}
                            aria-current={active ? 'true' : undefined}
                            className={`grid w-full grid-cols-[minmax(120px,1.4fr)_repeat(4,minmax(88px,1fr))] items-center gap-2 rounded-2xl px-3 py-2 text-left transition-colors ${
                              active ? 'bg-surface-2' : 'hover:bg-surface-2/70'}`}>
                            <span role="cell" className="font-medium">{r.city}</span>
                            {cells.map((v, i) => (
                              <span role="cell" key={i}
                                className="tabular flex h-12 items-center justify-between rounded-xl px-3 font-mono text-lg font-medium"
                                style={{ background: colorFor(v), color: inkOn(v), opacity: i === 0 ? 1 : 0.92 }}>
                                {v}
                                {i > 0 && <span className="text-[11px] font-sans font-normal opacity-75">fcst</span>}
                              </span>
                            ))}
                          </button>
                        </Reveal>
                      )
                    })}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
