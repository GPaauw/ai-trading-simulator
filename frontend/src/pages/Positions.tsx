import { useEffect, useState } from 'react'
import { api } from '../api'

interface Holding {
  symbol: string
  quantity: number
  avg_price: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
}

function formatEur(n: number) {
  return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR' }).format(n)
}

export default function Positions() {
  const [holdings, setHoldings] = useState<Holding[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getHoldings().then(setHoldings).catch(console.error).finally(() => setLoading(false))
    const interval = setInterval(() => {
      api.getHoldings().then(setHoldings).catch(console.error)
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="text-dark-400 text-center py-20">Laden...</div>

  if (holdings.length === 0) {
    return (
      <div className="bg-dark-900 rounded-xl border border-dark-800 p-10 text-center">
        <p className="text-dark-400 text-lg">Geen open posities</p>
        <p className="text-dark-500 text-sm mt-2">De AI zoekt naar kansen...</p>
      </div>
    )
  }

  const totalValue = holdings.reduce((s, h) => s + h.market_value, 0)
  const totalPnl = holdings.reduce((s, h) => s + h.unrealized_pnl, 0)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{holdings.length} Open Posities</h2>
        <div className="text-sm text-dark-400">
          Waarde: {formatEur(totalValue)} &middot;{' '}
          <span className={totalPnl >= 0 ? 'text-accent-green' : 'text-accent-red'}>
            P&L: {formatEur(totalPnl)}
          </span>
        </div>
      </div>

      <div className="bg-dark-900 rounded-xl border border-dark-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-dark-800 text-dark-400">
              <th className="text-left px-4 py-3 font-medium">Symbool</th>
              <th className="text-right px-4 py-3 font-medium">Aantal</th>
              <th className="text-right px-4 py-3 font-medium">Gem. Prijs</th>
              <th className="text-right px-4 py-3 font-medium">Huidig</th>
              <th className="text-right px-4 py-3 font-medium">Waarde</th>
              <th className="text-right px-4 py-3 font-medium">P&L</th>
              <th className="text-right px-4 py-3 font-medium">P&L %</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h) => (
              <tr key={h.symbol} className="border-b border-dark-800/50 hover:bg-dark-800/30">
                <td className="px-4 py-3 font-semibold">{h.symbol}</td>
                <td className="px-4 py-3 text-right">{h.quantity.toFixed(4)}</td>
                <td className="px-4 py-3 text-right">{formatEur(h.avg_price)}</td>
                <td className="px-4 py-3 text-right">{formatEur(h.current_price)}</td>
                <td className="px-4 py-3 text-right">{formatEur(h.market_value)}</td>
                <td className={`px-4 py-3 text-right font-medium ${
                  h.unrealized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'
                }`}>
                  {formatEur(h.unrealized_pnl)}
                </td>
                <td className={`px-4 py-3 text-right ${
                  h.unrealized_pnl_pct >= 0 ? 'text-accent-green' : 'text-accent-red'
                }`}>
                  {h.unrealized_pnl_pct >= 0 ? '+' : ''}{h.unrealized_pnl_pct.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
