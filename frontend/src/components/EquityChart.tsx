import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface Props {
  data: { date: string; equity: number }[]
}

export default function EquityChart({ data }: Props) {
  if (data.length === 0) return null

  const startEquity = data[0].equity
  const isPositive = data[data.length - 1].equity >= startEquity

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={isPositive ? '#22c55e' : '#ef4444'} stopOpacity={0.3} />
            <stop offset="95%" stopColor={isPositive ? '#22c55e' : '#ef4444'} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="date"
          tick={{ fill: '#85889d', fontSize: 11 }}
          tickFormatter={(v) => {
            const d = new Date(v)
            return `${d.getDate()}/${d.getMonth() + 1}`
          }}
          minTickGap={40}
        />
        <YAxis
          tick={{ fill: '#85889d', fontSize: 11 }}
          tickFormatter={(v) => `€${v.toLocaleString()}`}
          width={80}
          domain={['auto', 'auto']}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e1f2e', border: '1px solid #434557', borderRadius: 8 }}
          labelStyle={{ color: '#fff' }}
          formatter={(value: number) => [`€${value.toFixed(2)}`, 'Vermogen']}
          labelFormatter={(label) => new Date(label).toLocaleDateString('nl-NL')}
        />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={isPositive ? '#22c55e' : '#ef4444'}
          fill="url(#equityGrad)"
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
