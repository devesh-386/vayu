import { useEffect, useState } from 'react'

export type Outlook = {
  horizon: number
  date: string
  aqi: number
  low: number
  high: number
  category: string
  model: string
  alert: boolean
  possible_alert: boolean
}

export type CityData = {
  city: string
  updated: string
  today: {
    date: string
    aqi: number
    category: string
    dominant: string
    advice: string
    pollutants: Record<string, { value: number; sub_index: number }>
    weather: { temp: number; humidity: number; wind_speed: number; precip: number }
  }
  outlook: Outlook[]
  history: { date: string; aqi: number }[]
  why: { factor: string; contribution: number }[]
}

export type OverviewRow =
  | { city: string; today: { date: string; aqi: number; category: string; dominant: string }; outlook: Outlook[] }
  | { city: string; error: string }

export type Metrics = {
  MAE: number
  RMSE: number
  R2: number
  category_accuracy: number
  category_within_one: number
  alert_precision: number
  alert_recall: number
}

export type ModelReport = {
  generated: string
  data: { cities: string[]; start: string; end: string; city_days: number; source: string }
  split: Record<'train' | 'validation' | 'test', [string, string, number]>
  best_model: string
  results: Record<string, Metrics>
  horizons: Record<string, { best_model: string; interval: { coverage_80: number; mean_width: number }; test_mae: Record<string, number> }>
  test_mae_by_city: Record<string, Record<string, number>>
  shap_top: Record<string, number>
  features: { all: number; selected: number }
}

export const CITIES = ['Chennai', 'Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Lucknow']

type State<T> = { data: T | null; error: string | null; loading: boolean }

export function useApi<T>(path: string | null, retryKey = 0): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true })
  useEffect(() => {
    if (!path) return
    const ctrl = new AbortController()
    setState(s => ({ ...s, loading: true, error: null }))
    fetch(path, { signal: ctrl.signal })
      .then(async r => {
        if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `HTTP ${r.status}`)
        return r.json() as Promise<T>
      })
      .then(data => setState({ data, error: null, loading: false }))
      .catch(e => {
        if (e.name !== 'AbortError') setState({ data: null, error: String(e.message ?? e), loading: false })
      })
    return () => ctrl.abort()
  }, [path, retryKey])
  return state
}
