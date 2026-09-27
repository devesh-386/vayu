import { gsap } from 'gsap'
import { useLayoutEffect, useRef, type ReactNode } from 'react'

/** Fades content up once when it enters the viewport. Static under reduced motion. */
export function Reveal({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    gsap.set(el, { opacity: 0, y: 24 })
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return
      gsap.to(el, { opacity: 1, y: 0, duration: 0.9, delay, ease: 'expo.out' })
      io.disconnect()
    }, { threshold: 0.15 })
    io.observe(el)
    return () => { io.disconnect(); gsap.killTweensOf(el); gsap.set(el, { clearProps: 'all' }) }
  }, [delay])

  return <div ref={ref} className={className}>{children}</div>
}
