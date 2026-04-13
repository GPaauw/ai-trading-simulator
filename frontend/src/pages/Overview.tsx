import { useEffect, useState } from 'react'
import { api } from '../api'
import EquityChart from '../components/EquityChart'

interface Summary {
  total_equity: number
  cash: number
  holdings_value: number
  total_pnl: number
  total_pnl_pct: number
  start_balance: number
  num_holdings: number
  day_pnl: number
}

interface StatusInfo {
  auto_trader_running: boolean
  model_loaded: boolean
  instruments_count: number
  uptime_seconds: number
}

function formatEur(n: number) {
  return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR' }).format(n)
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-dark-900 rounded-xl p-5 border border-dark-800">
      <p className="text-dark-400 text-sm mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color || 'text-white'}`}>{value}</p>
      {sub && <p className={`text-sm mt-1 ${color || 'text-dark-400'}`}>{sub}</p>}
    </div>
  )
}

export default function Overview() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [status, setStatus] = useState<StatusInfo | null>(null)
  const [history, setHistory] = useState<{ date: string; equity: number }[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const [s, st, h] = await Promise.all([
          api.getPortfolioSummary(),
          api.getStatus(),
          api.getPortfolioHistory(),
        ])
        setSummary(s)
        setStatus(st)
        setHistory(h)
      } catch (e) {
        console.error('Load failed:', e)
      } finally {
        setLoading(false)
      }
    }
    load()
    const interval = setInterval(load, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <div className="text-dark-400 text-center py-20">Laden...</div>
  }

  if (!summary) {
    return <div className="text-red-400 text-center py-20">Kon data niet laden</div>
  }

  const pnlColor = summary.total_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'
  const dayColor = summary.day_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="flex items-center gap-3 text-sm">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full ${
          status?.auto_trader_running ? 'bg-green-900/30 text-accent-green' : 'bg-red-900/30 text-accent-red'
        }`}>
          <span className={`w-2 h-2 rounded-full ${status?.auto_trader_running ? 'bg-accent-green' : 'bg-accent-red'}`} />
          Auto-trader {status?.auto_trader_running ? 'actief' : 'gestopt'}
        </span>
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full ${
          status?.model_loaded ? 'bg-blue-900/30 text-accent-blue' : 'bg-yellow-900/30 text-accent-yellow'
        }`}>
          ML Model {status?.model_loaded ? 'geladen' : 'niet geladen (fallback)'}
        </span>
        <span className="text-dark-500">
          {status?.instruments_count} instrumenten
        </span>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Totaal Vermogen" value={formatEur(summary.total_equity)} />
        <StatCard
          label="Totaal P&L"
          value={formatEur(summary.total_pnl)}
          sub={`${summary.total_pnl_pct >= 0 ? '+' : ''}${summary.total_pnl_pct.toFixed(2)}%`}
          color={pnlColor}
        />
        <StatCard
          label="Dag P&L"
          value={formatEur(summary.day_pnl)}
          color={dayColor}
        />
        <StatCard label="Cash" value={formatEur(summary.cash)} sub={`${summary.num_holdings} posities`} />
      </div>

      {/* Equity chart */}
      <div className="bg-dark-900 rounded-xl border border-dark-800 p-5">
        <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
        {history.length > 0 ? (
          <EquityChart data={history} />
        ) : (
          <p className="text-dark-500 text-center py-10">Nog geen historie beschikbaar</p>
        )}
      </div>
    </div>
  )
}
