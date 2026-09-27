import { useState } from 'react'
import { CityBoard } from './components/CityBoard'
import { Footer } from './components/Footer'
import { Forecast } from './components/Forecast'
import { Hero } from './components/Hero'
import { Limits } from './components/Limits'
import { ModelProof } from './components/ModelProof'
import { Nav } from './components/Nav'
import { Pipeline } from './components/Pipeline'
import { useApi, type CityData, type ModelReport, type OverviewRow } from './lib/api'

export default function App() {
  const [city, setCity] = useState('Chennai')
  const [retry, setRetry] = useState(0)
  const cityState = useApi<CityData>(`/api/city/${city}`)
  const overview = useApi<OverviewRow[]>('/api/overview', retry)
  const model = useApi<ModelReport>('/api/model')

  return (
    <>
      <a href="#forecast" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-full focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-ink">
        Skip to forecast
      </a>
      <Nav />
      <main>
        <Hero city={city} onCity={setCity} data={cityState.data} loading={cityState.loading} error={cityState.error} />
        <Forecast city={city} data={cityState.data} loading={cityState.loading} error={cityState.error} />
        <CityBoard rows={overview.data} loading={overview.loading} error={overview.error}
          selected={city} onSelect={setCity} onRetry={() => setRetry(r => r + 1)} />
        <ModelProof report={model.data} />
        <Pipeline />
        <Limits />
      </main>
      <Footer />
    </>
  )
}
