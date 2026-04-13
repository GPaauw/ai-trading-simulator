import { useEffect, useState } from 'react'
import { api } from '../api'

interface Trade {
  id: number
  symbol: string
  side: string
  quantity: number
  price: number
  total: number
  pnl: number
  confidence: number
  timestamp: string
}

function formatEur(n: number) {
  return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR' }).format(n)
}

export default function Trades() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'buy' | 'sell'>('all')

  useEffect(() => {
    api.getTradeHistory().then(setTrades).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-dark-400 text-center py-20">Laden...</div>

  const filtered = filter === 'all' ? trades : trades.filter((t) => t.side === filter)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Trade Geschiedenis</h2>
        <div className="flex gap-1 bg-dark-900 rounded-lg p-1 border border-dark-800">
          {(['all', 'buy', 'sell'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded text-sm transition ${
                filter === f ? 'bg-dark-700 text-white' : 'text-dark-400 hover:text-white'
              }`}
            >
              {f === 'all' ? 'Alles' : f === 'buy' ? 'Koop' : 'Verkoop'}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="bg-dark-900 rounded-xl border border-dark-800 p-10 text-center text-dark-400">
          Geen trades gevonden
        </div>
      ) : (
        <div className="bg-dark-900 rounded-xl border border-dark-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-800 text-dark-400">
                <th className="text-left px-4 py-3 font-medium">Tijd</th>
                <th className="text-left px-4 py-3 font-medium">Symbool</th>
                <th className="text-left px-4 py-3 font-medium">Type</th>
                <th className="text-right px-4 py-3 font-medium">Aantal</th>
                <th className="text-right px-4 py-3 font-medium">Prijs</th>
                <th className="text-right px-4 py-3 font-medium">Totaal</th>
                <th className="text-right px-4 py-3 font-medium">P&L</th>
                <th className="text-right px-4 py-3 font-medium">Conf.</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => (
                <tr key={t.id} className="border-b border-dark-800/50 hover:bg-dark-800/30">
                  <td className="px-4 py-3 text-dark-400">
                    {new Date(t.timestamp).toLocaleString('nl-NL', {
                      day: '2-digit',
                      month: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td className="px-4 py-3 font-semibold">{t.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      t.side === 'buy'
                        ? 'bg-green-900/30 text-accent-green'
                        : 'bg-red-900/30 text-accent-red'
                    }`}>
                      {t.side === 'buy' ? 'KOOP' : 'VERKOOP'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">{t.quantity.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right">{formatEur(t.price)}</td>
                  <td className="px-4 py-3 text-right">{formatEur(t.total)}</td>
                  <td className={`px-4 py-3 text-right font-medium ${
                    t.pnl > 0 ? 'text-accent-green' : t.pnl < 0 ? 'text-accent-red' : 'text-dark-400'
                  }`}>
                    {t.pnl !== 0 ? formatEur(t.pnl) : '—'}
                  </td>
                  <td className="px-4 py-3 text-right text-dark-400">
                    {(t.confidence * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
