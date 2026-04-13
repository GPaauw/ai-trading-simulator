import { useState } from 'react'
import Overview from './Overview'
import Positions from './Positions'
import Trades from './Trades'
import AIInsight from './AIInsight'
import Management from './Management'

const tabs = [
  { id: 'overview', label: '📊 Overzicht' },
  { id: 'positions', label: '💼 Posities' },
  { id: 'trades', label: '📋 Trades' },
  { id: 'ai', label: '🤖 AI Insight' },
  { id: 'manage', label: '⚙️ Beheer' },
] as const

type TabId = (typeof tabs)[number]['id']

export default function Dashboard({ onLogout }: { onLogout: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>('overview')

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="bg-dark-900 border-b border-dark-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold">📈 AI Trading Simulator</h1>
        <div className="flex items-center gap-4">
          <span className="text-sm text-dark-400">v3.0</span>
          <button
            onClick={onLogout}
            className="text-sm text-dark-400 hover:text-white transition"
          >
            Uitloggen
          </button>
        </div>
      </header>

      {/* Tabs */}
      <nav className="bg-dark-900 border-b border-dark-800 px-6 flex gap-1 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition border-b-2 ${
              activeTab === tab.id
                ? 'border-accent-blue text-white'
                : 'border-transparent text-dark-400 hover:text-dark-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="p-6 max-w-7xl mx-auto">
        {activeTab === 'overview' && <Overview />}
        {activeTab === 'positions' && <Positions />}
        {activeTab === 'trades' && <Trades />}
        {activeTab === 'ai' && <AIInsight />}
        {activeTab === 'manage' && <Management />}
      </main>
    </div>
  )
}
