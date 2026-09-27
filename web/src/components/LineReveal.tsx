// Adapted from Animmaster: Hero Animations/15 (TextLinesReveal).
// Lines are split with split-type, wrapped in overflow masks and slid up in sequence.
// Changes: splits only after web fonts load (line breaks depend on the font), and restores
// the original markup when done so the text reflows normally on resize.
// Reduced motion: text is shown immediately.
import { gsap } from 'gsap'
import { useLayoutEffect, useRef, type ElementType, type ReactNode } from 'react'
import SplitType from 'split-type'

type Props = { as?: ElementType; className?: string; children: ReactNode; delay?: number }

export function LineReveal({ as: Tag = 'h1', className, children, delay = 0 }: Props) {
  const ref = useRef<HTMLElement>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let split: SplitType | null = null
    let tween: gsap.core.Tween | null = null
    let cancelled = false
    gsap.set(el, { autoAlpha: 0 })

    document.fonts.ready.then(() => {
      if (cancelled) return
      split = new SplitType(el, { types: 'lines' })
      split.lines?.forEach(line => {
        const mask = document.createElement('div')
        mask.className = 'line-mask'
        line.parentNode?.insertBefore(mask, line)
        mask.appendChild(line)
      })
      gsap.set(el, { autoAlpha: 1 })
      tween = gsap.fromTo(split.lines, { yPercent: 105 }, {
        yPercent: 0, duration: 1.4, ease: 'power4.out', stagger: 0.08, delay,
        onComplete: () => { split?.revert(); split = null },
      })
    })

    return () => {
      cancelled = true
      tween?.kill()
      split?.revert()
      gsap.set(el, { clearProps: 'opacity,visibility' })
    }
  }, [delay])

  return <Tag ref={ref} className={className}>{children}</Tag>
}
