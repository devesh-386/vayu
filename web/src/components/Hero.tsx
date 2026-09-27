import { ArrowDownIcon, ArrowRightIcon } from '@phosphor-icons/react'
import { useMemo } from 'react'
import { CITIES, type CityData } from '../lib/api'
import { band, colorFor, dayLabel } from '../lib/aqi'
import { LineReveal } from './LineReveal'
import { ParticleField } from './ParticleField'

type Props = {
  city: string
  onCity: (c: string) => void
  data: CityData | null
  loading: boolean
  error: string | null
}

export function Hero({ city, onCity, data, loading, error }: Props) {
  const aqi = data?.today.aqi ?? 80
  const colors = useMemo(() => [colorFor(aqi), '#8a919c', '#b9c0ca'], [aqi])
  const tomorrow = data?.outlook[0]

  return (
    <section id="top" className="relative isolate overflow-hidden">
      {/* Particle haze: denser and tinted by the selected city's live AQI. */}
      <ParticleField aqi={aqi} colors={colors}
        className="pointer-events-none absolute inset-0 -z-10 opacity-90 [mask-image:radial-gradient(ellipse_75%_70%_at_70%_45%,black,transparent)]" />

      <div className="mx-auto grid min-h-[calc(100dvh-4rem)] max-w-[1320px] items-center gap-12 px-5 pb-16 pt-12 md:px-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:pt-16">
        <div>
          <LineReveal className="text-5xl font-semibold leading-[1.02] tracking-[-0.035em] md:text-6xl xl:text-7xl">
            Tomorrow's air,<br /> forecast today.
          </LineReveal>
          <p className="mt-6 max-w-[46ch] text-lg leading-relaxed text-muted">
            Machine learning forecasts of the Air Quality Index for eight Indian cities, up to three days ahead.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <a href="#forecast"
               className="group inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3.5 font-medium text-accent-ink transition-transform active:scale-[0.98]">
              See the forecast
              <ArrowDownIcon size={18} weight="bold" className="transition-transform group-hover:translate-y-0.5" />
            </a>
            <a href="#method"
               className="group inline-flex items-center gap-2 rounded-full border border-line px-6 py-3.5 font-medium transition-colors hover:bg-surface-2 active:scale-[0.98]">
              How it works
              <ArrowRightIcon size={18} className="transition-transform group-hover:translate-x-0.5" />
            </a>
          </div>
        </div>

        <div className="panel relative p-6 backdrop-blur-sm md:p-8 [background:color-mix(in_srgb,var(--surface)_82%,transparent)]">
          <div className="-mx-1 flex flex-wrap gap-1" role="tablist" aria-label="City">
            {CITIES.map(c => (
              <button key={c} role="tab" aria-selected={c === city} onClick={() => onCity(c)}
                className={`shrink-0 rounded-full px-3.5 py-1.5 text-sm transition-colors ${
                  c === city ? 'bg-ink text-bg' : 'text-muted hover:bg-surface-2 hover:text-ink'}`}>
                {c}
              </button>
            ))}
          </div>

          <div className="mt-8 min-h-[248px]" aria-live="polite">
            {error ? (
              <p className="text-muted">Live data is unavailable right now. {error}</p>
            ) : loading || !data ? (
              <div className="space-y-4">
                <div className="skeleton h-5 w-40" />
                <div className="skeleton h-28 w-56" />
                <div className="skeleton h-5 w-64" />
              </div>
            ) : (
              <>
                <p className="text-sm text-muted">AQI now in {data.city}</p>
                <div className="mt-1 flex items-end gap-4">
                  <span className="tabular font-mono text-[7.5rem] font-medium leading-[0.9] tracking-[-0.05em]">
                    {data.today.aqi}
                  </span>
                  <span className="mb-3 inline-flex items-center gap-2 text-lg font-medium">
                    <span className="h-3 w-8 rounded-full" style={{ background: colorFor(data.today.aqi) }} aria-hidden />
                    {data.today.category}
                  </span>
                </div>
                {tomorrow && (
                  <div className="mt-8 grid grid-cols-3 gap-2 border-t border-line pt-5">
                    {data.outlook.map(o => (
                      <div key={o.date}>
                        <p className="text-xs text-muted">{dayLabel(o.date, data.today.date)}</p>
                        <p className="tabular mt-1 font-mono text-2xl font-medium">{o.aqi}</p>
                        <span className="mt-1.5 block h-1 w-10 rounded-full" style={{ background: colorFor(o.aqi) }} aria-hidden />
                        <p className="mt-1.5 text-xs text-muted">{band(o.aqi).name}</p>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
