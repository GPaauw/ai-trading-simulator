import { useState } from 'react'
import { api } from '../api'

function formatEur(n: number) {
  return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR' }).format(n)
}

export default function Management() {
  const [amount, setAmount] = useState('')
  const [message, setMessage] = useState<{ text: string; type: 'ok' | 'err' } | null>(null)
  const [loading, setLoading] = useState(false)

  const handleTransaction = async (type: 'deposit' | 'withdraw') => {
    const val = parseFloat(amount)
    if (isNaN(val) || val <= 0) {
      setMessage({ text: 'Voer een geldig bedrag in', type: 'err' })
      return
    }
    setLoading(true)
    setMessage(null)
    try {
      const res = type === 'deposit' ? await api.deposit(val) : await api.withdraw(val)
      setMessage({ text: `${res.message} — Nieuw saldo: ${formatEur(res.new_balance)}`, type: 'ok' })
      setAmount('')
    } catch (e) {
      setMessage({ text: `Fout: ${e instanceof Error ? e.message : 'Onbekend'}`, type: 'err' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Storten / Opnemen */}
      <div className="bg-dark-900 rounded-xl border border-dark-800 p-6">
        <h2 className="text-lg font-semibold mb-4">💰 Storten / Opnemen</h2>

        {message && (
          <div className={`mb-4 px-4 py-2 rounded text-sm ${
            message.type === 'ok'
              ? 'bg-green-900/30 border border-green-700 text-green-300'
              : 'bg-red-900/30 border border-red-700 text-red-300'
          }`}>
            {message.text}
          </div>
        )}

        <div className="flex gap-3">
          <div className="relative flex-1">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-dark-400">€</span>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="Bedrag"
              min="0"
              step="100"
              className="w-full bg-dark-800 border border-dark-700 rounded-lg pl-8 pr-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-accent-blue"
            />
          </div>
          <button
            onClick={() => handleTransaction('deposit')}
            disabled={loading}
            className="px-5 py-2.5 bg-accent-green hover:bg-green-600 disabled:opacity-50 text-white font-medium rounded-lg transition"
          >
            Storten
          </button>
          <button
            onClick={() => handleTransaction('withdraw')}
            disabled={loading}
            className="px-5 py-2.5 bg-accent-red hover:bg-red-600 disabled:opacity-50 text-white font-medium rounded-lg transition"
          >
            Opnemen
          </button>
        </div>
      </div>

      {/* Info */}
      <div className="bg-dark-900 rounded-xl border border-dark-800 p-6">
        <h2 className="text-lg font-semibold mb-4">ℹ️ Systeem Info</h2>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-dark-400">Versie</span>
            <span>v3.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">ML Model</span>
            <span>LightGBM + PPO RL</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Markten</span>
            <span>Stocks, Crypto, Commodities</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Strategie</span>
            <span>Automatisch (day/swing)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Training</span>
            <span>Wekelijks via GitHub Actions</span>
          </div>
          <div className="flex justify-between">
            <span className="text-dark-400">Transactiekosten</span>
            <span>0.1% + 0.05% slippage</span>
          </div>
        </div>
      </div>
    </div>
  )
}
