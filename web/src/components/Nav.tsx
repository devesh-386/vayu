const LINKS = [
  { href: '#forecast', label: 'Forecast' },
  { href: '#cities', label: 'Cities' },
  { href: '#model', label: 'Accuracy' },
  { href: '#method', label: 'Method' },
]

export function Nav() {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/80 backdrop-blur-md">
      <nav className="mx-auto flex h-16 max-w-[1320px] items-center justify-between px-5 md:px-8" aria-label="Main">
        <a href="#top" className="flex items-baseline gap-2 font-semibold tracking-tight">
          <span className="text-lg">Vayu</span>
          <span className="hidden text-sm font-normal text-muted sm:inline">AQI forecasts</span>
        </a>
        <ul className="flex items-center gap-1 text-sm">
          {LINKS.map(l => (
            <li key={l.href}>
              <a href={l.href}
                 className="rounded-full px-3 py-2 text-muted transition-colors hover:bg-surface-2 hover:text-ink">
                {l.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  )
}
