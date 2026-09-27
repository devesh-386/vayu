// CPCB National AQI bands. These colours encode data, so they stay fixed across themes.
export const BANDS = [
  { name: 'Good', max: 50, color: '#1f9d6b' },
  { name: 'Satisfactory', max: 100, color: '#8bbd3c' },
  { name: 'Moderate', max: 200, color: '#e8c21c' },
  { name: 'Poor', max: 300, color: '#f0892b' },
  { name: 'Very Poor', max: 400, color: '#df3b3b' },
  { name: 'Severe', max: 500, color: '#8e1537' },
] as const

export function band(aqi: number | null | undefined) {
  if (aqi == null || Number.isNaN(aqi)) return BANDS[0]
  return BANDS.find(b => aqi <= b.max) ?? BANDS[BANDS.length - 1]
}

export const colorFor = (aqi: number | null | undefined) => band(aqi).color

/** Text colour that stays readable on top of a band colour. */
export const inkOn = (aqi: number | null | undefined) =>
  ['Satisfactory', 'Moderate', 'Poor'].includes(band(aqi).name) ? '#16130a' : '#ffffff'

export const POLLUTANT_LABEL: Record<string, string> = {
  pm2_5: 'PM2.5', pm10: 'PM10', no2: 'NO₂', so2: 'SO₂', co: 'CO', o3: 'O₃',
}

export const POLLUTANT_UNIT: Record<string, string> = {
  pm2_5: 'µg/m³', pm10: 'µg/m³', no2: 'µg/m³', so2: 'µg/m³', co: 'mg/m³', o3: 'µg/m³',
}

export function dayLabel(iso: string, today: string) {
  const d = (Date.parse(iso) - Date.parse(today)) / 86_400_000
  if (d === 0) return 'Today'
  if (d === 1) return 'Tomorrow'
  return new Date(iso).toLocaleDateString('en-IN', { weekday: 'short' })
}

export const shortDate = (iso: string) =>
  new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
