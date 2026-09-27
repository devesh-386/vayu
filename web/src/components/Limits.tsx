import { Reveal } from './Reveal'

const ITEMS = [
  { title: 'Modelled air, not street monitors',
    body: 'Pollutant history comes from the CAMS atmospheric model on a 40 km grid, which reads ozone high over India. CPCB station data would give more realistic absolute values.' },
  { title: 'One reading per city',
    body: 'Each city is a single grid point, so the forecast describes the city as a whole, not a neighbourhood or a busy junction.' },
  { title: 'Severe days are rare in training',
    body: 'Only one Severe day appears before 2025, so extreme episodes are forecast less reliably than ordinary ones.' },
]

export function Limits() {
  return (
    <section className="mx-auto max-w-[1320px] px-5 py-24 md:px-8 md:py-32">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <Reveal>
          <h2 className="max-w-[14ch] text-3xl font-semibold tracking-[-0.025em] md:text-5xl">What it cannot tell you yet.</h2>
        </Reveal>
        <div className="space-y-10">
          {ITEMS.map((it, i) => (
            <Reveal key={it.title} delay={i * 0.06}>
              <h3 className="text-xl font-medium">{it.title}</h3>
              <p className="mt-2 max-w-[62ch] leading-relaxed text-muted">{it.body}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}
