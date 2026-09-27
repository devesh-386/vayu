export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto grid max-w-[1320px] gap-10 px-5 py-14 text-sm md:grid-cols-3 md:px-8">
        <div>
          <p className="text-base font-semibold">Vayu</p>
          <p className="mt-2 max-w-[36ch] text-muted">
            Intelligent Air Quality Prediction System. 21CSC305P Machine Learning, SRM Institute of Science and Technology, Ramapuram.
          </p>
        </div>
        <div>
          <p className="font-medium">Team</p>
          <ul className="mt-2 space-y-1 text-muted">
            <li>Hasini S</li>
            <li>Roshni Rajakumari M P</li>
            <li>Navashri N M</li>
          </ul>
          <p className="mt-4 font-medium">Supervisor</p>
          <p className="mt-1 text-muted">Dr. M. B. Sudhan</p>
        </div>
        <div className="text-muted">
          <p className="font-medium text-ink">Data</p>
          <p className="mt-2">
            Contains modified Copernicus Atmosphere Monitoring Service information. Weather from ERA5 and forecast models via{' '}
            <a className="underline decoration-line underline-offset-4 hover:text-ink" href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a>{' '}
            (CC BY 4.0). AQI follows the CPCB National Air Quality Index.
          </p>
          <p className="mt-4">Forecasts are experimental and not an official air quality advisory.</p>
        </div>
      </div>
    </footer>
  )
}
