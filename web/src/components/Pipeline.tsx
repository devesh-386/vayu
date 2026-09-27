// Horizontal pan built from the design-taste canonical GSAP skeleton (pin at "top top",
// scrub the inner track). Desktop with motion only; otherwise a plain vertical list.
import { BellIcon, BrainIcon, BroomIcon, ChartLineUpIcon, DatabaseIcon, FunctionIcon, FunnelIcon } from '@phosphor-icons/react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useLayoutEffect, useRef } from 'react'

gsap.registerPlugin(ScrollTrigger)

const STAGES = [
  { icon: DatabaseIcon, title: 'Collect', stat: '12,048', unit: 'city-days',
    body: 'Hourly PM2.5, PM10, NO₂, SO₂, CO and O₃ from the CAMS model, with weather from ERA5, for eight cities since August 2022.' },
  { icon: BroomIcon, title: 'Clean', stat: '3', unit: 'day gap limit',
    body: 'Daily values follow CPCB averaging rules. Short gaps are interpolated and outliers clipped with limits learned from training data only.' },
  { icon: FunctionIcon, title: 'Engineer', stat: '55', unit: 'candidate features',
    body: 'Pollution lags and trends, the weather forecast for the target day, season, Diwali timing and crop-burning season.' },
  { icon: FunnelIcon, title: 'Select', stat: '34', unit: 'features kept',
    body: 'Permutation importance on held-out data drops anything that does not reduce forecast error.' },
  { icon: BrainIcon, title: 'Train', stat: '4', unit: 'models compared',
    body: 'Linear regression, Random Forest, XGBoost and an LSTM, each tuned on the first half of 2025 and tested on the months after.' },
  { icon: ChartLineUpIcon, title: 'Forecast', stat: '1-3', unit: 'days ahead',
    body: 'A CPCB AQI and category for each day, an 80% range calibrated on validation data, and a SHAP breakdown of the reasons.' },
  { icon: BellIcon, title: 'Warn', stat: '>200', unit: 'AQI triggers an alert',
    body: 'Days forecast Poor or worse raise a warning on the dashboard, with optional email alerts.' },
]

export function Pipeline() {
  const wrap = useRef<HTMLElement>(null)
  const track = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const mm = gsap.matchMedia()
    mm.add('(min-width: 1024px) and (prefers-reduced-motion: no-preference)', () => {
      const t = track.current!
      const distance = () => t.scrollWidth - window.innerWidth
      gsap.to(t, {
        x: () => -distance(),
        ease: 'none',
        scrollTrigger: {
          trigger: wrap.current,
          start: 'top top',
          end: () => `+=${distance()}`,
          pin: true,
          scrub: 1,
          invalidateOnRefresh: true,
        },
      })
    })
    return () => mm.revert()
  }, [])

  return (
    <section id="method" ref={wrap} className="relative overflow-hidden">
      <div className="flex min-h-[100dvh] flex-col justify-center py-24 lg:py-0">
        <div className="mx-auto w-full max-w-[1320px] px-5 md:px-8">
          <h2 className="max-w-[20ch] text-3xl font-semibold tracking-[-0.025em] md:text-5xl">From raw readings to a warning.</h2>
          <p className="mt-4 max-w-[60ch] text-muted">
            The same pipeline runs for each forecast horizon. Nothing from the test period is used to build it.
          </p>
        </div>
        <div ref={track} className="mt-12 flex flex-col gap-4 px-5 md:px-8 lg:w-max lg:flex-row lg:gap-5 lg:pl-[max(2rem,calc((100vw-1320px)/2+2rem))] lg:pr-[20vw]">
          {STAGES.map(({ icon: Icon, title, stat, unit, body }) => (
            <article key={title} className="panel flex flex-col p-6 lg:h-[340px] lg:w-[340px] lg:p-7">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-semibold tracking-tight">{title}</h3>
                <Icon size={26} className="text-accent" />
              </div>
              <p className="mt-4 leading-relaxed text-muted">{body}</p>
              <p className="mt-auto pt-6">
                <span className="tabular font-mono text-3xl font-medium">{stat}</span>
                <span className="ml-2 text-sm text-muted">{unit}</span>
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
