// Hand-rolled (no cloud/rain component in the Animmaster pack).
// Soft drifting clouds and falling rain drawn on a 2D canvas, driven by the selected city's
// real weather: humidity sets cloud cover, rainfall sets rain density, wind sets drift and slant.
// Pauses off-screen; reduced motion draws one still frame.
import { useEffect, useRef } from 'react'

type Props = { humidity: number; precip: number; wind: number; className?: string }

type Cloud = { x: number; y: number; r: number; speed: number; puffs: { dx: number; dy: number; r: number }[] }
type Drop = { x: number; y: number; len: number; speed: number }

export function WeatherSky({ humidity, precip, wind, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let w = 0, h = 0

    const cover = Math.min(1, Math.max(0.15, (humidity - 30) / 60 + precip / 10))
    const drift = 0.15 + Math.min(wind, 40) / 40 * 0.6
    const slant = Math.min(wind, 40) / 40 * 0.35
    const dropCount = precip <= 0.05 ? 0 : Math.round(Math.min(40 + precip * 45, 420))

    let clouds: Cloud[] = []
    let drops: Drop[] = []
    const makeCloud = (x: number): Cloud => {
      const r = 60 + Math.random() * 90
      return {
        x, y: h * (0.05 + Math.random() * 0.4), r, speed: drift * (0.5 + Math.random() * 0.8),
        puffs: Array.from({ length: 5 + Math.floor(Math.random() * 4) }, () => ({
          dx: (Math.random() - 0.5) * r * 2.4, dy: (Math.random() - 0.5) * r * 0.5, r: r * (0.45 + Math.random() * 0.5),
        })),
      }
    }
    const makeDrop = (y?: number): Drop => ({
      x: Math.random() * (w + 200) - 100, y: y ?? Math.random() * h,
      len: 10 + Math.random() * 16, speed: 7 + Math.random() * 6,
    })

    const resize = () => {
      w = canvas.clientWidth; h = canvas.clientHeight
      canvas.width = w * dpr; canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const n = Math.round(3 + cover * 7)
      clouds = Array.from({ length: n }, (_, i) => makeCloud((i / n) * (w + 400) - 200))
      drops = Array.from({ length: dropCount }, () => makeDrop())
    }
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    resize()

    const cloudRgb = dark ? '170,180,196' : '120,130,148'
    const cloudAlpha = (dark ? 0.07 : 0.1) * (0.6 + cover * 0.6)
    const rainRgb = dark ? '150,175,220' : '70,95,150'

    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      for (const c of clouds) {
        for (const p of c.puffs) {
          const g = ctx.createRadialGradient(c.x + p.dx, c.y + p.dy, 0, c.x + p.dx, c.y + p.dy, p.r)
          g.addColorStop(0, `rgba(${cloudRgb},${cloudAlpha})`)
          g.addColorStop(1, `rgba(${cloudRgb},0)`)
          ctx.fillStyle = g
          ctx.beginPath(); ctx.arc(c.x + p.dx, c.y + p.dy, p.r, 0, Math.PI * 2); ctx.fill()
        }
      }
      ctx.lineWidth = 1
      ctx.lineCap = 'round'
      for (const d of drops) {
        const a = 0.18 + (d.speed - 7) / 6 * 0.2
        ctx.strokeStyle = `rgba(${rainRgb},${a})`
        ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - d.len * slant, d.y + d.len); ctx.stroke()
      }
    }
    const step = (k: number) => {
      for (const c of clouds) {
        c.x += c.speed * k
        if (c.x - c.r * 2 > w) Object.assign(c, makeCloud(-c.r * 2.5))
      }
      for (const d of drops) {
        d.y += d.speed * k; d.x -= d.speed * slant * k
        if (d.y > h) Object.assign(d, makeDrop(-d.len))
      }
    }

    let raf = 0, last = performance.now(), visible = true
    const tick = (t: number) => {
      raf = requestAnimationFrame(tick)
      const k = Math.min((t - last) / 16.7, 3); last = t
      step(k); draw()
    }
    const start = () => { if (!raf && visible && !reduce) { last = performance.now(); raf = requestAnimationFrame(tick) } }
    const stop = () => { cancelAnimationFrame(raf); raf = 0 }
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) start(); else stop() })
    io.observe(canvas)
    if (reduce) draw(); else start()

    return () => { stop(); io.disconnect(); ro.disconnect() }
  }, [humidity, precip, wind])

  return <canvas ref={ref} aria-hidden className={className} />
}
