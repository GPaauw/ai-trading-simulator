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
  bool _loadingSignals = false;
  bool _loadingHistory = false;
  bool _loadingLearn = false;
  Map<String, dynamic>? _learnResult;
  String? _signalError;

  @override
  void initState() {
    super.initState();
    _refreshSignals();
    _refreshHistory();
  }

  Future<void> _refreshSignals() async {
    setState(() {
      _loadingSignals = true;
      _signalError = null;
    });
    try {
      final signals = await ApiClient.getSignals();
      if (mounted) setState(() => _signals = signals);
    } catch (e) {
      if (mounted) setState(() => _signalError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingSignals = false);
    }
  }

  Future<void> _refreshHistory() async {
    setState(() => _loadingHistory = true);
    try {
      final history = await ApiClient.getHistory();
      if (mounted) setState(() => _history = history);
    } catch (_) {
      // stil falen — gebruiker kan handmatig vernieuwen
    } finally {
      if (mounted) setState(() => _loadingHistory = false);
    }
  }

  Future<void> _executeTrade(Signal signal) async {
    try {
      final trade =
          await ApiClient.executeTrade(signal.symbol, signal.action, 1000.0);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Trade: ${trade.symbol} ${trade.action.toUpperCase()} — '
            'P/L: \$${trade.profitLoss.toStringAsFixed(2)} | ${trade.status}',
          ),
          backgroundColor:
              trade.profitLoss >= 0 ? Colors.green[700] : Colors.red[700],
          duration: const Duration(seconds: 4),
        ),
      );
      _refreshHistory();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Fout: $e'),
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
      if (mounted) setState(() => _loadingLearn = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Trading Simulator'),
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
        onRefresh: () async {
          await Future.wait([_refreshSignals(), _refreshHistory()]);
        },
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 900),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildSignalsCard(),
                  const SizedBox(height: 20),
                  _buildLearnCard(),
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

  // ── Signalen ──────────────────────────────────────────────────────────────

  Widget _buildSignalsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Signalen',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: 'Vernieuwen',
                  onPressed: _loadingSignals ? null : _refreshSignals,
                ),
              ],
            ),
            const Divider(),
            if (_loadingSignals)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (_signalError != null)
              Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  _signalError!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              )
            else if (_signals.isEmpty)
              const Padding(
                padding: EdgeInsets.all(12),
                child: Text('Geen signalen beschikbaar.'),
              )
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
    final isBuy = signal.action == 'buy';
    final color = isBuy ? Colors.green : Colors.red;
    return ListTile(
      leading: CircleAvatar(
        backgroundColor: color,
        child: Text(
          isBuy ? '↑' : '↓',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      title: Text(
        '${signal.symbol}  —  \$${signal.price.toStringAsFixed(2)}',
        style: const TextStyle(fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        '${signal.action.toUpperCase()}  |  '
        'Zekerheid: ${(signal.confidence * 100).toStringAsFixed(0)}%',
      ),
      trailing: FilledButton(
        style: FilledButton.styleFrom(backgroundColor: color),
        onPressed: () => _executeTrade(signal),
        child: const Text('Execute'),
      ),
    );
  }

  // ── Leermodule ────────────────────────────────────────────────────────────

  Widget _buildLearnCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('AI Leermodule', style: Theme.of(context).textTheme.titleLarge),
            const Divider(),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                icon: _loadingLearn
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.psychology),
                label: const Text('Leer van Trades'),
                onPressed: _loadingLearn ? null : _learn,
              ),
            ),
            if (_learnResult != null) ...[
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.indigo.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.indigo.withOpacity(0.3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Leerresultaten',
                      style: Theme.of(context)
                          .textTheme
                          .bodyLarge
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    ..._learnResult!.entries.map(
                      (e) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2),
                        child: Row(
                          children: [
                            Text(
                              '${e.key}: ',
                              style: const TextStyle(fontWeight: FontWeight.w500),
                            ),
                            Text('${e.value}'),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ── Trade Geschiedenis ────────────────────────────────────────────────────

  Widget _buildHistoryCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Trade Geschiedenis',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: 'Vernieuwen',
                  onPressed: _loadingHistory ? null : _refreshHistory,
                ),
              ],
            ),
            const Divider(),
            if (_loadingHistory)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (_history.isEmpty)
              const Padding(
                padding: EdgeInsets.all(12),
                child: Text('Nog geen trades uitgevoerd.'),
              )
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _history.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) {
                  // Nieuwste bovenaan
                  final trade = _history[_history.length - 1 - i];
                  return _buildTradeTile(trade);
                },
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTradeTile(Trade trade) {
    final isProfit = trade.profitLoss >= 0;
    final color = isProfit ? Colors.green : Colors.red;
    final timestamp = trade.timestamp.length >= 19
        ? trade.timestamp.substring(0, 19).replaceFirst('T', ' ')
        : trade.timestamp;

    return ListTile(
      leading: Icon(
        isProfit ? Icons.trending_up : Icons.trending_down,
        color: color,
      ),
      title: Text(
        '${trade.symbol}  —  ${trade.action.toUpperCase()}  '
        '(€${trade.amount.toStringAsFixed(0)})',
      ),
      subtitle: Text('$timestamp  |  ${trade.status}'),
      trailing: Text(
        '${isProfit ? '+' : ''}\$${trade.profitLoss.toStringAsFixed(2)}',
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.bold,
          fontSize: 15,
        ),
      ),
    );
  }
}
