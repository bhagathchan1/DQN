import React from 'react'

export const numberOf = (...values: unknown[]): number | undefined => {
  for (const value of values) if (typeof value === 'number' && Number.isFinite(value)) return value
  return undefined
}
export const fmt = (value: unknown, digits = 2) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—'

export async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = payload.detail ?? payload.error ?? `${response.status} ${response.statusText}`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return payload as T
}

export function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    play: <path d="m9 5 11 7-11 7Z" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
    arrow: <><path d="M5 12h14m-5-5 5 5-5 5" /></>,
    chart: <><path d="M4 4v16h17M8 14l4-5 4 3 5-7" /></>,
    history: <><path d="M3 11a9 9 0 1 1 2 7M3 5v6h6m3-4v6l4 2" /></>,
    game: <><path d="M7 8h10a4 4 0 0 1 4 4l1 4a3 3 0 0 1-5 2l-2-2H9l-2 2a3 3 0 0 1-5-2l1-4a4 4 0 0 1 4-4Z" /><path d="M7 10v5m-2-2h4m7-1h.01m3 2h.01" /></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    brain: <><path d="M12 3v18M12 6C8 0 2 5 5 9c-5 4-1 11 4 9 0 3 3 3 3 0m0-12c4-6 10-1 7 3 5 4 1 11-4 9 0 3-3 3-3 0" /><path d="m6 8 3 2m9-2-3 2M7 15l2-2m8 2-2-2" /></>,
    download: <><path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5" /></>,
    refresh: <><path d="M20 11a8 8 0 0 0-14-5L3 9m0-6v6h6m-5 4a8 8 0 0 0 14 5l3-3m0 6v-6h-6" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10h.01" /></>,
    check: <path d="m5 12 4 4L20 5" />,
    plus: <path d="M12 5v14M5 12h14" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.info}</svg>
}
