import { useState } from 'react'
import { api } from '../api'

export default function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await api.login(username, password)
      api.setToken(res.token)
      onLogin()
    } catch {
      setError('Login mislukt. Controleer je gegevens.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-dark-950">
      <form
        onSubmit={handleSubmit}
        className="bg-dark-900 p-8 rounded-xl shadow-2xl w-full max-w-sm space-y-6"
      >
        <h1 className="text-2xl font-bold text-center">📈 AI Trading Simulator</h1>
        <p className="text-dark-400 text-center text-sm">Log in om verder te gaan</p>

        {error && (
          <div className="bg-red-900/30 border border-red-700 text-red-300 px-4 py-2 rounded text-sm">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm text-dark-300 mb-1">Gebruikersnaam</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full bg-dark-800 border border-dark-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-blue"
            required
            autoComplete="username"
          />
        </div>

        <div>
          <label className="block text-sm text-dark-300 mb-1">Wachtwoord</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-dark-800 border border-dark-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-blue"
            required
            autoComplete="current-password"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-accent-blue hover:bg-blue-600 disabled:opacity-50 text-white font-semibold py-2.5 rounded-lg transition"
        >
          {loading ? 'Bezig...' : 'Inloggen'}
        </button>
      </form>
    </div>
  )
}
