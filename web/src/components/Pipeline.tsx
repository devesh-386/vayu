// Adapted from Animmaster: Scroll Animation/33 (scroll-scrubbed stroke drawing).
// A vertical line draws itself as the reader scrolls; each stage lights up when the line
// reaches it. No pinning, so the section never overlaps its neighbours.
// Reduced motion: line fully drawn, all stages active.
import { BellIcon, BrainIcon, BroomIcon, ChartLineUpIcon, DatabaseIcon, FunctionIcon, FunnelIcon } from '@phosphor-icons/react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useLayoutEffect, useRef } from 'react'

gsap.registerPlugin(ScrollTrigger)

const STAGES = [
  { icon: DatabaseIcon, title: 'Collect', stat: '12,048', unit: 'city-days',
    body: 'Hourly PM2.5, PM10, NO₂, SO₂, CO and O₃ from the CAMS model, with weather from ERA5, for eight cities since August 2022.' },
  { icon: BroomIcon, title: 'Clean', stat: '48', unit: 'outlier readings clipped',
    body: 'Daily values follow CPCB averaging rules. Short gaps are interpolated and outliers clipped with limits learned from training data only.' },
  { icon: FunctionIcon, title: 'Engineer', stat: '55', unit: 'candidate features',
    body: 'Pollution lags and trends, the weather forecast for the target day, season, Diwali timing and crop-burning season.' },
  { icon: FunnelIcon, title: 'Select', stat: '34', unit: 'features kept',
    body: 'Permutation importance on held-out data drops anything that does not reduce forecast error.' },
  { icon: BrainIcon, title: 'Train', stat: '4', unit: 'models compared',
    body: 'Linear regression, Random Forest, XGBoost and an LSTM, each tuned on the first half of 2025 and tested on the months after.' },
  { icon: ChartLineUpIcon, title: 'Forecast', stat: '1-3', unit: 'days ahead',
    body: 'A CPCB AQI and category for each day, an 80% range calibrated on validation data, and a SHAP breakdown of the reasons.' },
  { icon: BellIcon, title: 'Warn', stat: '>200', unit: 'AQI raises an alert',
    body: 'Days forecast Poor or worse raise a warning on the dashboard, with optional email alerts.' },
]

export function Pipeline() {
  const root = useRef<HTMLElement>(null)
  const line = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const el = root.current
    if (!el) return
    const steps = gsap.utils.toArray<HTMLElement>('.stage', el)
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      steps.forEach(s => s.classList.add('is-on'))
      return
    }
    const ctx = gsap.context(() => {
      gsap.fromTo(line.current, { scaleY: 0 }, {
        scaleY: 1, ease: 'none', transformOrigin: 'top center',
        scrollTrigger: { trigger: '.stage-list', start: 'top 60%', end: 'bottom 60%', scrub: 0.6 },
      })
      steps.forEach(s => {
        ScrollTrigger.create({ trigger: s, start: 'top 60%', toggleClass: 'is-on' })
      })
    }, el)
    return () => ctx.revert()
  }, [])

  return (
    <section id="method" ref={root} className="mx-auto max-w-[1320px] px-5 py-24 md:px-8 md:py-32">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="lg:sticky lg:top-28 lg:self-start">
          <h2 className="max-w-[16ch] text-3xl font-semibold tracking-[-0.025em] md:text-5xl">From raw readings to a warning.</h2>
          <p className="mt-4 max-w-[46ch] text-muted">
            The same pipeline runs for each forecast horizon. Nothing from the test period is used to build it.
          </p>
        </div>

        <ol className="stage-list relative">
          <div className="absolute bottom-6 left-[19px] top-6 w-px bg-line" aria-hidden />
          <div ref={line} className="absolute bottom-6 left-[19px] top-6 w-px origin-top bg-accent" aria-hidden />
          {STAGES.map(({ icon: Icon, title, stat, unit, body }) => (
            <li key={title} className="stage group relative grid grid-cols-[40px_minmax(0,1fr)] gap-5 pb-12 last:pb-0">
              <span className="relative z-10 grid h-10 w-10 place-items-center rounded-full border border-line bg-bg text-faint transition-colors duration-500 group-[.is-on]:border-accent group-[.is-on]:text-accent">
                <Icon size={18} />
              </span>
              <div className="pt-1.5 opacity-40 transition-opacity duration-500 group-[.is-on]:opacity-100">
                <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
                  <h3 className="text-2xl font-semibold tracking-tight">{title}</h3>
                  <p><span className="tabular font-mono text-2xl font-medium">{stat}</span>
                    <span className="ml-2 text-sm text-muted">{unit}</span></p>
                </div>
                <p className="mt-2 max-w-[56ch] leading-relaxed text-muted">{body}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
