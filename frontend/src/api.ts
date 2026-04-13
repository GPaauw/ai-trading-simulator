const BASE = import.meta.env.VITE_API_URL || '';

let token: string | null = localStorage.getItem('token');

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    token = null;
    localStorage.removeItem('token');
    window.location.reload();
    throw new Error('Unauthorized');
  }

  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }

  return res.json();
}

export const api = {
  setToken(t: string) {
    token = t;
    localStorage.setItem('token', t);
  },

  clearToken() {
    token = null;
    localStorage.removeItem('token');
  },

  isAuthenticated() {
    return !!token;
  },

  login(username: string, password: string) {
    return request<{ token: string }>('/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  },

  getStatus() {
    return request<{
      status: string;
      version: string;
      auto_trader_running: boolean;
      model_loaded: boolean;
      instruments_count: number;
      uptime_seconds: number;
    }>('/status');
  },

  getPortfolioSummary() {
    return request<{
      total_equity: number;
      cash: number;
      holdings_value: number;
      total_pnl: number;
      total_pnl_pct: number;
      start_balance: number;
      num_holdings: number;
      day_pnl: number;
    }>('/portfolio/summary');
  },

  getHoldings() {
    return request<{
      symbol: string;
      quantity: number;
      avg_price: number;
      current_price: number;
      market_value: number;
      unrealized_pnl: number;
      unrealized_pnl_pct: number;
    }[]>('/portfolio/holdings');
  },

  getPortfolioHistory() {
    return request<{
      date: string;
      equity: number;
      cash: number;
      holdings_value: number;
    }[]>('/portfolio/history');
  },

  getTradeHistory() {
    return request<{
      id: number;
      symbol: string;
      side: string;
      quantity: number;
      price: number;
      total: number;
      pnl: number;
      confidence: number;
      timestamp: string;
    }[]>('/trades/history');
  },

  getSignals() {
    return request<{
      symbol: string;
      action: string;
      confidence: number;
      expected_return: number;
      risk_score: number;
    }[]>('/ai/signals');
  },

  getInsights() {
    return request<{
      model_type: string;
      accuracy: number;
      feature_importance: Record<string, number>;
      recent_decisions: number;
      win_rate: number;
    }>('/ai/insights');
  },

  deposit(amount: number) {
    return request<{ message: string; new_balance: number }>('/portfolio/deposit', {
      method: 'POST',
      body: JSON.stringify({ amount }),
    });
  },

  withdraw(amount: number) {
    return request<{ message: string; new_balance: number }>('/portfolio/withdraw', {
      method: 'POST',
      body: JSON.stringify({ amount }),
    });
  },

  updateSettings(settings: Record<string, unknown>) {
    return request<{ message: string }>('/settings/update', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  },
};
