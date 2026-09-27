import { CloudRainIcon, DropIcon, ThermometerIcon, WarningIcon, WindIcon } from '@phosphor-icons/react'
import type { CityData } from '../lib/api'
import { POLLUTANT_LABEL, POLLUTANT_UNIT, colorFor, dayLabel } from '../lib/aqi'
import { ForecastChart } from './ForecastChart'
import { Reveal } from './Reveal'

type Props = { data: CityData | null; loading: boolean; error: string | null; city: string }

export function Forecast({ data, loading, error, city }: Props) {
  return (
    <section id="forecast" className="mx-auto max-w-[1320px] px-5 py-24 md:px-8 md:py-32">
      <Reveal>
        <h2 className="text-3xl font-semibold tracking-[-0.025em] md:text-5xl">{city}, the next three days.</h2>
      </Reveal>
      {error ? (
        <p className="panel mt-10 p-6 text-muted">The forecast for {city} could not be produced: {error}</p>
      ) : loading || !data ? (
        <div className="mt-10 grid gap-4 lg:grid-cols-3">
          <div className="skeleton h-[380px] lg:col-span-2" />
          <div className="skeleton h-[380px]" />
        </div>
      ) : (
        <Body data={data} />
      )}
    </section>
  )
}

function Body({ data }: { data: CityData }) {
  const worst = data.outlook.reduce((a, b) => (b.aqi > a.aqi ? b : a))
  const watch = data.outlook.find(o => o.possible_alert)
  const maxAbs = Math.max(...data.why.map(w => Math.abs(w.contribution)), 1)
  const pollutants = Object.entries(data.today.pollutants).sort((a, b) => b[1].sub_index - a[1].sub_index)

  return (
    <>
      {(worst.alert || watch) && (
        <div role="status" className="panel mt-8 flex items-start gap-3 p-4 md:p-5"
          style={{ borderColor: colorFor(worst.alert ? worst.aqi : 250) }}>
          <WarningIcon size={22} weight="fill" style={{ color: colorFor(worst.alert ? worst.aqi : 250) }} className="mt-0.5 shrink-0" />
          <p>
            {worst.alert
              ? <><strong>Early warning.</strong> {dayLabel(worst.date, data.today.date)} is forecast at AQI {worst.aqi}, {worst.category}. Limit time outdoors.</>
              : <><strong>Worth watching.</strong> The upper end of the {dayLabel(watch!.date, data.today.date).toLowerCase()} range reaches AQI {watch!.high}, which is Poor. Most likely value is {watch!.aqi}.</>}
          </p>
        </div>
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Reveal className="panel p-5 md:p-6 lg:col-span-2">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="font-medium">Observed and forecast AQI</h3>
            <p className="text-sm text-muted">Shaded band: 80% forecast range</p>
          </div>
          <div className="mt-4">
            <ForecastChart data={data} />
          </div>
        </Reveal>

        <Reveal className="panel p-5 md:p-6" delay={0.08}>
          <h3 className="font-medium">Why tomorrow looks like this</h3>
          <p className="mt-1 text-sm text-muted">AQI points each factor adds or removes compared with today (SHAP).</p>
          <ul className="mt-6 space-y-4">
            {data.why.map(w => (
              <li key={w.factor}>
                <div className="flex justify-between text-sm">
                  <span>{w.factor}</span>
                  <span className="tabular font-mono">{w.contribution > 0 ? '+' : ''}{w.contribution.toFixed(0)}</span>
                </div>
                <div className="mt-1.5 grid grid-cols-2 gap-px">
                  <div className="flex justify-end">
                    {w.contribution < 0 && <span className="h-2 rounded-l-full bg-[#1f9d6b]" style={{ width: `${(Math.abs(w.contribution) / maxAbs) * 100}%` }} />}
                  </div>
                  <div>
                    {w.contribution > 0 && <span className="block h-2 rounded-r-full bg-[#df3b3b]" style={{ width: `${(w.contribution / maxAbs) * 100}%` }} />}
                  </div>
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-6 text-xs text-muted">Green lowers the forecast, red raises it.</p>
        </Reveal>

        <Reveal className="panel p-5 md:p-6">
          <h3 className="font-medium">Conditions today</h3>
          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-5">
            {[
              { icon: ThermometerIcon, label: 'Temperature', value: `${data.today.weather.temp}°C` },
              { icon: DropIcon, label: 'Humidity', value: `${data.today.weather.humidity}%` },
              { icon: WindIcon, label: 'Wind', value: `${data.today.weather.wind_speed} km/h` },
              { icon: CloudRainIcon, label: 'Rain', value: `${data.today.weather.precip} mm` },
            ].map(({ icon: Icon, label, value }) => (
              <div key={label}>
                <dt className="flex items-center gap-1.5 text-xs text-muted"><Icon size={14} /> {label}</dt>
                <dd className="tabular mt-1 font-mono text-xl">{value}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-6 border-t border-line pt-5 text-sm leading-relaxed text-muted">{data.today.advice}</p>
        </Reveal>

        <Reveal className="panel p-5 md:p-6 lg:col-span-2" delay={0.08}>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="font-medium">What drives today's AQI</h3>
            <p className="text-sm text-muted">The AQI is the highest pollutant sub-index</p>
          </div>
          <div className="mt-6 grid gap-x-8 gap-y-5 sm:grid-cols-2">
            {pollutants.map(([p, v], i) => (
              <div key={p}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className={i === 0 ? 'font-medium' : ''}>{POLLUTANT_LABEL[p]}</span>
                  <span className="tabular font-mono text-sm text-muted">{v.value} {POLLUTANT_UNIT[p]}</span>
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <span className="h-2 rounded-full" style={{ width: `${Math.max(4, (v.sub_index / Math.max(pollutants[0][1].sub_index, 100)) * 85)}%`, background: colorFor(v.sub_index) }} />
                  <span className="tabular font-mono text-sm">{v.sub_index}</span>
                </div>
              </div>
            ))}
          </div>
        </Reveal>
      </div>
      <p className="mt-4 text-xs text-muted">
        Forecasts by {data.outlook[0].model}. Updated {new Date(data.updated).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}.
      </p>
    </>
  )
}
