import { useEffect, useState } from 'react'
import { api } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface Insight {
  model_type: string
  accuracy: number
  feature_importance: Record<string, number>
  recent_decisions: number
  win_rate: number
}

interface Signal {
  symbol: string
  action: string
  confidence: number
  expected_return: number
  risk_score: number
}

const actionColors: Record<string, string> = {
  strong_buy: '#22c55e',
  buy: '#4ade80',
  hold: '#6b7280',
  sell: '#f87171',
  strong_sell: '#ef4444',
}

export default function AIInsight() {
  const [insight, setInsight] = useState<Insight | null>(null)
  const [signals, setSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.getInsights(), api.getSignals()])
      .then(([i, s]) => {
        setInsight(i)
        setSignals(s)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-dark-400 text-center py-20">Laden...</div>

  // Feature importance chart data
  const fiData = insight?.feature_importance
    ? Object.entries(insight.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15)
        .map(([name, value]) => ({ name, value: Math.round(value * 100) / 100 }))
    : []

  return (
    <div className="space-y-6">
      {/* Model Info */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-dark-900 rounded-xl p-5 border border-dark-800">
          <p className="text-dark-400 text-sm mb-1">Model</p>
          <p className="text-lg font-bold">{insight?.model_type || 'Onbekend'}</p>
        </div>
        <div className="bg-dark-900 rounded-xl p-5 border border-dark-800">
          <p className="text-dark-400 text-sm mb-1">Nauwkeurigheid</p>
          <p className="text-lg font-bold text-accent-blue">
            {insight?.accuracy ? `${(insight.accuracy * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-dark-900 rounded-xl p-5 border border-dark-800">
          <p className="text-dark-400 text-sm mb-1">Win Rate</p>
          <p className="text-lg font-bold text-accent-green">
            {insight?.win_rate ? `${(insight.win_rate * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-dark-900 rounded-xl p-5 border border-dark-800">
          <p className="text-dark-400 text-sm mb-1">Recente Beslissingen</p>
          <p className="text-lg font-bold">{insight?.recent_decisions ?? 0}</p>
        </div>
      </div>

      {/* Signals */}
      <div className="bg-dark-900 rounded-xl border border-dark-800 p-5">
        <h2 className="text-lg font-semibold mb-4">Huidige Signalen</h2>
        {signals.length === 0 ? (
          <p className="text-dark-500 text-center py-6">Geen actieve signalen</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {signals.slice(0, 12).map((s) => (
              <div
                key={s.symbol}
                className="flex items-center justify-between bg-dark-800/50 rounded-lg px-4 py-3"
              >
                <div>
                  <span className="font-semibold">{s.symbol}</span>
                  <span
                    className="ml-2 px-2 py-0.5 rounded text-xs font-medium"
                    style={{
                      backgroundColor: `${actionColors[s.action] || '#6b7280'}20`,
                      color: actionColors[s.action] || '#6b7280',
                    }}
                  >
                    {s.action.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                <div className="text-right text-sm">
                  <div className="text-dark-300">{(s.confidence * 100).toFixed(0)}% conf</div>
                  <div className={s.expected_return >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                    {s.expected_return >= 0 ? '+' : ''}{(s.expected_return * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Feature Importance */}
      {fiData.length > 0 && (
        <div className="bg-dark-900 rounded-xl border border-dark-800 p-5">
          <h2 className="text-lg font-semibold mb-4">Feature Importance (Top 15)</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={fiData} layout="vertical" margin={{ left: 120 }}>
              <XAxis type="number" tick={{ fill: '#85889d', fontSize: 12 }} />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#b0b2c0', fontSize: 12 }}
                width={110}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e1f2e', border: '1px solid #434557', borderRadius: 8 }}
                labelStyle={{ color: '#fff' }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {fiData.map((_, i) => (
                  <Cell key={i} fill={i < 5 ? '#3b82f6' : '#434557'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
