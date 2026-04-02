import 'package:flutter/material.dart';

import 'api_client.dart';
import 'login_page.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  Map<String, dynamic>? _summary;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _fetchSummary();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _fetchSummary() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await ApiClient.getAutoTraderSummary();
      if (!mounted) return;
      setState(() => _summary = data);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _logout() {
    ApiClient.clearToken();
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const LoginPage()),
    );
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  String _fmt(dynamic v, {int decimals = 2}) {
    if (v == null) return '-';
    final num n = v is num ? v : num.tryParse(v.toString()) ?? 0;
    return n.toStringAsFixed(decimals);
  }

  Color _plColor(dynamic v) {
    if (v == null) return Colors.white70;
    final num n = v is num ? v : num.tryParse(v.toString()) ?? 0;
    if (n > 0) return Colors.green;
    if (n < 0) return Colors.red;
    return Colors.white70;
  }

  // ── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Auto-Trader'),
        actions: [
          IconButton(
            icon: _loading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
            tooltip: 'Ververs',
            onPressed: _loading ? null : _fetchSummary,
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            tooltip: 'Uitloggen',
            onPressed: _logout,
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.dashboard), text: 'Status'),
            Tab(icon: Icon(Icons.account_balance_wallet), text: 'Open Posities'),
            Tab(icon: Icon(Icons.history), text: 'Trade-geschiedenis'),
          ],
        ),
      ),
      body: _loading && _summary == null
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _summary == null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.error_outline, size: 48, color: Colors.red),
                        const SizedBox(height: 12),
                        Text(_error!, textAlign: TextAlign.center),
                        const SizedBox(height: 16),
                        FilledButton.icon(
                          icon: const Icon(Icons.refresh),
                          label: const Text('Opnieuw proberen'),
                          onPressed: _fetchSummary,
                        ),
                      ],
                    ),
                  ),
                )
              : TabBarView(
                  controller: _tabController,
                  children: [
                    _buildStatusTab(),
                    _buildPositionsTab(),
                    _buildHistoryTab(),
                  ],
                ),
    );
  }

  // ── Tab 1: Status ────────────────────────────────────────────────────────

  Widget _buildStatusTab() {
    if (_summary == null) return const SizedBox.shrink();
    final s = _summary!;

    final running = s['running'] == true;
    final cycles = s['cycles_completed'] ?? 0;
    final interval = s['interval_minutes'] ?? '-';
    final lastCycle = s['last_cycle_time'] ?? '-';
    final serverTime = s['server_time'] ?? '-';
    final positions = (s['open_positions'] as List?)?.length ?? 0;
    final trades = (s['trade_history'] as List?)?.length ?? 0;
    final lastError = s['last_error'];

    // Portfolio
    final startBalance = s['start_balance'] ?? 0;
    final availableCash = s['available_cash'] ?? 0;
    final totalInvested = s['total_invested'] ?? 0;
    final totalMarketValue = s['total_market_value'] ?? 0;
    final totalEquity = s['total_equity'] ?? 0;
    final totalPnl = s['total_pnl'] ?? 0;
    final totalPnlPct = s['total_pnl_pct'] ?? 0;
    final unrealizedPnl = s['unrealized_pnl'] ?? 0;
    final realizedPnl = s['realized_pnl'] ?? 0;
    final totalBuys = s['total_buys'] ?? 0;
    final totalSells = s['total_sells'] ?? 0;

    return RefreshIndicator(
      onRefresh: _fetchSummary,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _statusCard(
            icon: running ? Icons.play_circle : Icons.pause_circle,
            iconColor: running ? Colors.green : Colors.orange,
            title: running ? 'Actief' : 'Gestopt',
            subtitle: 'Interval: $interval min  •  $serverTime (Amsterdam)',
          ),
          const SizedBox(height: 12),

          // Portfolio overzicht
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Portfolio', style: Theme.of(context).textTheme.titleMedium),
                  const Divider(),
                  _infoRow('Startkapitaal', '€ ${_fmt(startBalance)}'),
                  _infoRow('Beschikbaar cash', '€ ${_fmt(availableCash)}'),
                  _infoRow('Geïnvesteerd', '€ ${_fmt(totalInvested)}'),
                  _infoRow('Marktwaarde posities', '€ ${_fmt(totalMarketValue)}'),
                  const Divider(),
                  _infoRow('Totaal vermogen', '€ ${_fmt(totalEquity)}',
                      valueColor: Colors.white),
                  _infoRow(
                    'Totale winst / verlies',
                    '€ ${_fmt(totalPnl)} (${_fmt(totalPnlPct)}%)',
                    valueColor: _plColor(totalPnl),
                  ),
                  _infoRow(
                    'Ongerealiseerde P/L',
                    '€ ${_fmt(unrealizedPnl)}',
                    valueColor: _plColor(unrealizedPnl),
                  ),
                  _infoRow(
                    'Gerealiseerde P/L',
                    '€ ${_fmt(realizedPnl)}',
                    valueColor: _plColor(realizedPnl),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Trading activiteit
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Trading activiteit', style: Theme.of(context).textTheme.titleMedium),
                  const Divider(),
                  _infoRow('Cycli voltooid', '$cycles'),
                  _infoRow('Laatste cyclus', lastCycle),
                  _infoRow('Totaal buys', '$totalBuys'),
                  _infoRow('Totaal sells', '$totalSells'),
                  _infoRow('Open posities', '$positions'),
                  _infoRow('Totaal trades', '$trades'),
                ],
              ),
            ),
          ),
          if (lastError != null) ...[
            const SizedBox(height: 12),
            Card(
              color: Colors.red.withValues(alpha: 0.15),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const Icon(Icons.warning, color: Colors.red),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        lastError.toString(),
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _statusCard({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String subtitle,
  }) {
    return Card(
      child: ListTile(
        leading: Icon(icon, size: 40, color: iconColor),
        title: Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        subtitle: Text(subtitle),
      ),
    );
  }

  Widget _infoRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white70)),
          Text(value, style: TextStyle(fontWeight: FontWeight.w600, color: valueColor)),
        ],
      ),
    );
  }

  // ── Tab 2: Open Posities ─────────────────────────────────────────────────

  Widget _buildPositionsTab() {
    final positions = (_summary?['open_positions'] as List?) ?? [];

    if (positions.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.account_balance_wallet_outlined, size: 48, color: Colors.white38),
            SizedBox(height: 12),
            Text('Geen open posities', style: TextStyle(color: Colors.white54)),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _fetchSummary,
      child: ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: positions.length,
        itemBuilder: (context, index) {
          final p = positions[index] as Map<String, dynamic>;
          final pnl = p['unrealized_pnl'] ?? 0;
          final pnlPct = p['unrealized_pnl_pct'] ?? 0;
          return Card(
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '${p['symbol']}',
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: _plColor(pnl).withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '${_fmt(pnlPct)}%',
                          style: TextStyle(
                            color: _plColor(pnl),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _infoRow('Aantal', _fmt(p['quantity'], decimals: 4)),
                  _infoRow('Aankoopprijs', '€ ${_fmt(p['avg_entry_price'], decimals: 4)}'),
                  _infoRow('Huidige prijs', '€ ${_fmt(p['current_price'], decimals: 4)}'),
                  _infoRow('Geïnvesteerd', '€ ${_fmt(p['invested_amount'])}'),
                  _infoRow('Huidige waarde', '€ ${_fmt(p['current_value'])}'),
                  _infoRow(
                    'Ongerealiseerde P/L',
                    '€ ${_fmt(pnl)}',
                    valueColor: _plColor(pnl),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  // ── Tab 3: Trade-geschiedenis ────────────────────────────────────────────

  Widget _buildHistoryTab() {
    final trades = (_summary?['trade_history'] as List?) ?? [];

    if (trades.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.history, size: 48, color: Colors.white38),
            SizedBox(height: 12),
            Text('Nog geen trades', style: TextStyle(color: Colors.white54)),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _fetchSummary,
      child: ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: trades.length,
        itemBuilder: (context, index) {
          final t = trades[index] as Map<String, dynamic>;
          final action = (t['action'] ?? '').toString().toUpperCase();
          final isBuy = action == 'BUY';
          final pl = t['profit_loss'];
          return Card(
            child: ListTile(
              leading: CircleAvatar(
                backgroundColor: isBuy ? Colors.blue.withValues(alpha: 0.2) : Colors.orange.withValues(alpha: 0.2),
                child: Icon(
                  isBuy ? Icons.arrow_downward : Icons.arrow_upward,
                  color: isBuy ? Colors.blue : Colors.orange,
                ),
              ),
              title: Text(
                '${t['symbol']} — $action',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              subtitle: Text(
                '${t['timestamp'] ?? '-'}  •  € ${_fmt(t['price'], decimals: 4)}  •  ${_fmt(t['quantity'], decimals: 4)} stuks',
                style: const TextStyle(fontSize: 12),
              ),
              trailing: pl != null && pl != 0
                  ? Text(
                      '€ ${_fmt(pl)}',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: _plColor(pl),
                      ),
                    )
                  : null,
            ),
          );
        },
      ),
    );
  }
}
