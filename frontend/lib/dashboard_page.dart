import 'package:flutter/material.dart';

import 'api_client.dart';
import 'login_page.dart';
import 'models/signal.dart';
import 'models/trade.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  List<Signal> _signals = [];
  List<Trade> _history = [];
  Map<String, dynamic>? _learnResult;
  Map<String, dynamic>? _portfolio;

  bool _loadingSignals = false;
  bool _loadingHistory = false;
  bool _loadingLearn = false;
  bool _loadingPortfolio = false;

  String? _signalError;

  @override
  void initState() {
    super.initState();
    _refreshAll();
  }

  Future<void> _refreshAll() async {
    await Future.wait([
      _refreshSignals(),
      _refreshHistory(),
      _refreshPortfolio(),
    ]);
  }

  Future<void> _refreshSignals() async {
    setState(() {
      _loadingSignals = true;
      _signalError = null;
    });
    try {
      final signals = await ApiClient.getSignals();
      if (mounted) {
        setState(() => _signals = signals);
      }
    } catch (e) {
      if (mounted) {
        setState(() => _signalError = e.toString());
      }
    } finally {
      if (mounted) {
        setState(() => _loadingSignals = false);
      }
    }
  }

  Future<void> _refreshHistory() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingHistory = true);
    try {
      final history = await ApiClient.getHistory();
      if (mounted) {
        setState(() => _history = history);
      }
    } catch (_) {
      // Niet blokkeren; gebruiker kan handmatig vernieuwen.
    } finally {
      if (mounted) {
        setState(() => _loadingHistory = false);
      }
    }
  }

  Future<void> _refreshPortfolio() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingPortfolio = true);
    try {
      final portfolio = await ApiClient.getPortfolio();
      if (mounted) {
        setState(() => _portfolio = portfolio);
      }
    } catch (_) {
      // Niet blokkeren; gebruiker kan handmatig vernieuwen.
    } finally {
      if (mounted) {
        setState(() => _loadingPortfolio = false);
      }
    }
  }

  Future<void> _executePaperTrade(Signal signal) async {
    if (signal.action == 'hold') {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('HOLD-signaal kan niet worden uitgevoerd.')),
      );
      return;
    }

    final currentBalance = (_portfolio?['current_balance'] as num?)?.toDouble() ?? 2000.0;
    final suggestedAmount =
        (currentBalance * (signal.riskPct / 100.0) * 8.0).clamp(100.0, currentBalance * 0.4);

    try {
      final trade = await ApiClient.executeTrade(
        signal.symbol,
        signal.action,
        suggestedAmount,
        signal.expectedReturnPct,
        signal.riskPct,
      );

      if (!mounted) return;
      final isProfit = trade.profitLoss >= 0;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Paper trade ${trade.symbol} (${trade.action.toUpperCase()}) | '
            'P/L: ${trade.profitLoss.toStringAsFixed(2)}',
          ),
          backgroundColor: isProfit ? Colors.green[700] : Colors.red[700],
        ),
      );

      await Future.wait([_refreshHistory(), _refreshPortfolio()]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Tradefout: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
    }
  }

  Future<void> _learn() async {
    setState(() {
      _loadingLearn = true;
      _learnResult = null;
    });
    try {
      final result = await ApiClient.learn();
      if (mounted) {
        setState(() => _learnResult = result['parameters'] as Map<String, dynamic>?);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Leerfout: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _loadingLearn = false);
      }
    }
  }

  Future<void> _sendRealtimeAlerts() async {
    try {
      final result = await ApiClient.sendRealtimeAlerts();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Realtime alerts: ${result.toString()}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Alertfout: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
    }
  }

  Future<void> _sendDailySummary() async {
    try {
      final result = await ApiClient.sendDailySummary();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Dagelijkse samenvatting: ${result.toString()}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Summaryfout: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Market Advisor'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            tooltip: 'Uitloggen',
            onPressed: () {
              ApiClient.clearToken();
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (_) => const LoginPage()),
              );
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _refreshAll,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 980),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildPortfolioCard(),
                  const SizedBox(height: 20),
                  _buildAdviceCard(),
                  const SizedBox(height: 20),
                  _buildActionsCard(),
                  const SizedBox(height: 20),
                  _buildHistoryCard(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPortfolioCard() {
    final loggedIn = ApiClient.isLoggedIn();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Portfolio', style: Theme.of(context).textTheme.titleLarge),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _loadingPortfolio ? null : _refreshPortfolio,
                ),
              ],
            ),
            const Divider(),
            if (!loggedIn)
              const Text('Log in om je portfolio en trade limieten te bekijken.')
            else if (_loadingPortfolio)
              const Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(),
              )
            else if (_portfolio == null)
              const Text('Portfolio nog niet geladen.')
            else
              Wrap(
                spacing: 20,
                runSpacing: 8,
                children: [
                  Text('Start: €${(_portfolio!['start_balance'] as num).toStringAsFixed(2)}'),
                  Text('Nu: €${(_portfolio!['current_balance'] as num).toStringAsFixed(2)}'),
                  Text('P/L: ${(_portfolio!['total_profit_loss'] as num).toStringAsFixed(2)}'),
                  Text('Trades vandaag: ${_portfolio!['daily_trade_count']} / 10'),
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildAdviceCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Marktadvies (08:00-16:00)', style: Theme.of(context).textTheme.titleLarge),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _loadingSignals ? null : _refreshSignals,
                ),
              ],
            ),
            const Divider(),
            if (_loadingSignals)
              const Padding(
                padding: EdgeInsets.all(12),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_signalError != null)
              Text(_signalError!, style: TextStyle(color: Theme.of(context).colorScheme.error))
            else if (_signals.isEmpty)
              const Text('Geen adviezen beschikbaar.')
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _signals.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) => _buildSignalTile(_signals[i]),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildSignalTile(Signal signal) {
    final loggedIn = ApiClient.isLoggedIn();
    final actionColor = switch (signal.action) {
      'buy' => Colors.green,
      'sell' => Colors.red,
      _ => Colors.orange,
    };

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: actionColor,
        child: Text(
          signal.action == 'buy'
              ? '↑'
              : signal.action == 'sell'
                  ? '↓'
                  : '•',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      title: Text(
        '${signal.symbol} (${signal.market.toUpperCase()})  —  '
        '${signal.action.toUpperCase()} @ ${signal.price.toStringAsFixed(2)}',
      ),
      subtitle: Text(
        'Verwacht: ${signal.expectedReturnPct.toStringAsFixed(2)}% | '
        'Risico: ${signal.riskPct.toStringAsFixed(2)}% | '
        'Zekerheid: ${(signal.confidence * 100).toStringAsFixed(0)}%\n'
        '${signal.reason}',
      ),
      isThreeLine: true,
      trailing: loggedIn && signal.action != 'hold'
          ? FilledButton(
              style: FilledButton.styleFrom(backgroundColor: actionColor),
              onPressed: () => _executePaperTrade(signal),
              child: const Text('Paper trade'),
            )
          : const SizedBox.shrink(),
    );
  }

  Widget _buildActionsCard() {
    final loggedIn = ApiClient.isLoggedIn();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Leren & Alerts', style: Theme.of(context).textTheme.titleLarge),
            const Divider(),
            if (!loggedIn)
              const Text('Log in om leren en e-mailalerts te gebruiken.')
            else
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  FilledButton.icon(
                    icon: _loadingLearn
                        ? const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.psychology),
                    label: const Text('Leer van markt + trades'),
                    onPressed: _loadingLearn ? null : _learn,
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.notifications_active_outlined),
                    label: const Text('Stuur realtime alerts'),
                    onPressed: _sendRealtimeAlerts,
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(Icons.summarize_outlined),
                    label: const Text('Stuur dag samenvatting'),
                    onPressed: _sendDailySummary,
                  ),
                ],
              ),
            if (_learnResult != null) ...[
              const SizedBox(height: 14),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.indigo.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: _learnResult!.entries
                      .map((e) => Text('${e.key}: ${e.value}'))
                      .toList(),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryCard() {
    final loggedIn = ApiClient.isLoggedIn();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Paper Trade Geschiedenis', style: Theme.of(context).textTheme.titleLarge),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _loadingHistory ? null : _refreshHistory,
                ),
              ],
            ),
            const Divider(),
            if (!loggedIn)
              const Text('Log in om historie te bekijken.')
            else if (_loadingHistory)
              const Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(),
              )
            else if (_history.isEmpty)
              const Text('Nog geen paper trades uitgevoerd.')
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _history.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) {
                  final trade = _history[_history.length - 1 - i];
                  final isProfit = trade.profitLoss >= 0;
                  return ListTile(
                    leading: Icon(
                      isProfit ? Icons.trending_up : Icons.trending_down,
                      color: isProfit ? Colors.green : Colors.red,
                    ),
                    title: Text(
                      '${trade.symbol} ${trade.action.toUpperCase()} | '
                      '€${trade.amount.toStringAsFixed(0)}',
                    ),
                    subtitle: Text(
                      '${trade.timestamp.substring(0, 19).replaceFirst('T', ' ')} | '
                      'verwacht ${trade.expectedReturnPct.toStringAsFixed(2)}% | '
                      'risico ${trade.riskPct.toStringAsFixed(2)}% | ${trade.status}',
                    ),
                    trailing: Text(
                      '${isProfit ? '+' : ''}${trade.profitLoss.toStringAsFixed(2)}',
                      style: TextStyle(
                        color: isProfit ? Colors.green : Colors.red,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }
}
