import 'package:flutter/material.dart';

import 'api_client.dart';
import 'login_page.dart';
import 'models/holding.dart';
import 'models/signal.dart';
import 'models/trade.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  List<Signal> _signals = [];
  List<Holding> _holdings = [];
  List<Trade> _history = [];
  Map<String, dynamic>? _learnResult;
  Map<String, dynamic>? _portfolio;

  bool _loadingSignals = false;
  bool _loadingHoldings = false;
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
      _refreshHoldings(),
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

  Future<void> _refreshHoldings() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingHoldings = true);
    try {
      final holdings = await ApiClient.getSellAdvice();
      if (mounted) {
        setState(() => _holdings = holdings);
      }
    } catch (_) {
      // Niet blokkeren; gebruiker kan handmatig verversen.
    } finally {
      if (mounted) {
        setState(() => _loadingHoldings = false);
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

  Future<void> _registerBuyFromSignal(Signal signal) async {
    if (!ApiClient.isLoggedIn()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Log eerst in om een aankoop te registreren.')),
      );
      return;
    }

    final currentBalance = (_portfolio?['available_cash'] as num?)?.toDouble() ?? 2000.0;
    final suggestedAmount =
        (currentBalance * (signal.riskPct / 100.0) * 8.0).clamp(100.0, currentBalance * 0.4);
    final amountController = TextEditingController(
      text: suggestedAmount.toStringAsFixed(2),
    );
    final priceController = TextEditingController(text: signal.price.toStringAsFixed(4));
    final quantityController = TextEditingController();

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text('Aankoop registreren: ${signal.symbol}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: amountController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Totaal gekocht bedrag (€)'),
            ),
            TextField(
              controller: priceController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Aankoopprijs per stuk'),
            ),
            TextField(
              controller: quantityController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Aantal stuks (optioneel)'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Annuleren'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Opslaan'),
          ),
        ],
      ),
    );

    if (confirmed != true) {
      return;
    }

    final amount = double.tryParse(amountController.text.replaceAll(',', '.'));
    final price = double.tryParse(priceController.text.replaceAll(',', '.'));
    final quantity = quantityController.text.trim().isEmpty
        ? null
        : double.tryParse(quantityController.text.replaceAll(',', '.'));

    if (amount == null || amount <= 0 || price == null || price <= 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Vul een geldig bedrag en prijs in.')),
      );
      return;
    }

    try {
      final trade = await ApiClient.executeTrade(
        signal.symbol,
        'buy',
        amount,
        market: signal.market,
        quantity: quantity,
        price: price,
        expectedReturnPct: signal.expectedReturnPct,
        riskPct: signal.riskPct,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Aankoop opgeslagen: ${trade.symbol} | '
            '${trade.quantity.toStringAsFixed(4)} stuks @ ${trade.price.toStringAsFixed(2)}',
          ),
          backgroundColor: Colors.green[700],
        ),
      );

      await Future.wait([_refreshHistory(), _refreshPortfolio(), _refreshHoldings()]);
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

  Future<void> _sellAll() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Alle posities verkopen'),
        content: const Text(
          'Einde dag: wil je alle open posities sluiten?\n'
          'Dit verkoopt alles tegen de huidige marktprijs.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Annuleren'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Verkoop alles'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      final result = await ApiClient.sellAll();
      if (!mounted) return;
      final sold = result['sold'] as int;
      final pl = (result['total_profit_loss'] as num).toDouble();
      final isProfit = pl >= 0;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '$sold posities verkocht | '
            'Totaal P/L: €${pl.toStringAsFixed(2)}',
          ),
          backgroundColor: isProfit ? Colors.green[700] : Colors.red[700],
        ),
      );
      await Future.wait([_refreshHistory(), _refreshPortfolio(), _refreshHoldings()]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Fout bij verkopen: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
    }
  }

  Future<void> _sellHolding(Holding holding) async {
    final defaultQuantity = holding.suggestedSellFraction > 0
        ? holding.quantity * holding.suggestedSellFraction
        : holding.quantity;
    final quantityController = TextEditingController(text: defaultQuantity.toStringAsFixed(6));
    final priceController = TextEditingController(text: holding.currentPrice.toStringAsFixed(4));

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text('Verkoop ${holding.symbol}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: quantityController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Aantal te verkopen'),
            ),
            TextField(
              controller: priceController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Verkoopprijs per stuk'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Annuleren'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Verkoop'),
          ),
        ],
      ),
    );

    if (confirmed != true) {
      return;
    }

    final quantity = double.tryParse(quantityController.text.replaceAll(',', '.'));
    final price = double.tryParse(priceController.text.replaceAll(',', '.'));
    if (quantity == null || quantity <= 0 || price == null || price <= 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Vul een geldige hoeveelheid en prijs in.')),
      );
      return;
    }

    try {
      final trade = await ApiClient.executeTrade(
        holding.symbol,
        'sell',
        0,
        market: holding.market,
        quantity: quantity,
        price: price,
        expectedReturnPct: holding.expectedReturnPct,
        riskPct: holding.riskPct,
      );
      if (!mounted) return;
      final isProfit = trade.profitLoss >= 0;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Verkoop verwerkt: ${trade.symbol} | resultaat ${trade.profitLoss.toStringAsFixed(2)}',
          ),
          backgroundColor: isProfit ? Colors.green[700] : Colors.red[700],
        ),
      );
      await Future.wait([_refreshHistory(), _refreshPortfolio(), _refreshHoldings()]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Verkoopfout: $e'),
          backgroundColor: Colors.red[700],
        ),
      );
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
                  _buildHoldingsCard(),
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
                  Text('Cash: €${(_portfolio!['available_cash'] as num).toStringAsFixed(2)}'),
                  Text('Marktwaarde: €${(_portfolio!['market_value'] as num).toStringAsFixed(2)}'),
                  Text('P/L: ${(_portfolio!['total_profit_loss'] as num).toStringAsFixed(2)}'),
                  Text('Posities: ${_portfolio!['holdings_count']}'),
                  Text('Trades vandaag: ${_portfolio!['daily_trade_count']} / 50'),
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHoldingsCard() {
    final loggedIn = ApiClient.isLoggedIn();
    final stockHoldings = _holdings.where((holding) => holding.market != 'crypto').toList();
    final cryptoHoldings = _holdings.where((holding) => holding.market == 'crypto').toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Verkoopadvies voor jouw posities', style: Theme.of(context).textTheme.titleLarge),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (_holdings.isNotEmpty)
                      FilledButton.icon(
                        style: FilledButton.styleFrom(backgroundColor: Colors.red),
                        icon: const Icon(Icons.sell, size: 18),
                        label: const Text('Verkoop alles'),
                        onPressed: _sellAll,
                      ),
                    const SizedBox(width: 8),
                    IconButton(
                      icon: const Icon(Icons.refresh),
                      onPressed: _loadingHoldings ? null : _refreshHoldings,
                    ),
                  ],
                ),
              ],
            ),
            const Divider(),
            if (!loggedIn)
              const Text('Log in om verkoopadvies voor je bezittingen te bekijken.')
            else if (_loadingHoldings)
              const Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(),
              )
            else if (_holdings.isEmpty)
              const Text('Nog geen bezittingen geregistreerd.')
            else ...[
              _buildHoldingSection('Aandelen die je bezit', stockHoldings),
              const SizedBox(height: 16),
              _buildHoldingSection('Crypto die je bezit', cryptoHoldings),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildAdviceCard() {
    final stockSignals = _signals.where((signal) => signal.market != 'crypto').toList();
    final cryptoSignals = _signals.where((signal) => signal.market == 'crypto').toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Daytrades — koop vandaag, verkoop vandaag', style: Theme.of(context).textTheme.titleLarge),
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
              const Text('Er zijn nu geen daytrade-kansen volgens het model.')
            else ...[
              _buildSignalSection('Aandelen', stockSignals),
              const SizedBox(height: 16),
              _buildSignalSection('Crypto', cryptoSignals),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSignalSection(String title, List<Signal> signals) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        if (signals.isEmpty)
          Text('Geen daytrades beschikbaar voor $title.')
        else
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: signals.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, i) => _buildSignalTile(signals[i]),
          ),
      ],
    );
  }

  Widget _buildSignalTile(Signal signal) {
    final loggedIn = ApiClient.isLoggedIn();
    const actionColor = Colors.green;

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: actionColor,
        child: const Text('↑', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ),
      title: Text(
        '${signal.symbol} (${signal.market.toUpperCase()})  —  '
        'BUY @ ${signal.price.toStringAsFixed(2)}',
      ),
      subtitle: Text(
        '${signal.rankLabel.isEmpty ? 'Daytrade' : signal.rankLabel} | '
        'score ${signal.rankingScore.toStringAsFixed(1)}\n'
        'Verwacht: +${signal.expectedReturnPct.toStringAsFixed(2)}% → '
        'doel €${signal.targetPrice.toStringAsFixed(2)} | '
        'winst ~€${signal.expectedProfit.toStringAsFixed(0)} per €1000 | '
        'vandaag (daytrade)\n'
        'Stop-loss: ${signal.riskPct.toStringAsFixed(2)}% | '
        'Zekerheid: ${(signal.confidence * 100).toStringAsFixed(0)}%',
      ),
      isThreeLine: true,
      trailing: loggedIn
          ? FilledButton(
              style: FilledButton.styleFrom(backgroundColor: actionColor),
              onPressed: () => _registerBuyFromSignal(signal),
              child: const Text('Ik kocht dit'),
            )
          : const SizedBox.shrink(),
    );
  }

  Widget _buildHoldingSection(String title, List<Holding> holdings) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        if (holdings.isEmpty)
          Text('Geen posities beschikbaar in $title.')
        else
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: holdings.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, index) => _buildHoldingTile(holdings[index]),
          ),
      ],
    );
  }

  Widget _buildHoldingTile(Holding holding) {
    final pnlColor = holding.unrealizedProfitLoss >= 0 ? Colors.green : Colors.red;
    final buttonLabel = holding.recommendation == 'take_partial_profit'
        ? 'Neem winst'
        : 'Verkoop';

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: holding.recommendation == 'sell_now'
            ? Colors.red
            : holding.recommendation == 'take_partial_profit'
                ? Colors.orange
                : Colors.blueGrey,
        child: Text(
          holding.recommendation == 'sell_now'
              ? '↓'
              : holding.recommendation == 'take_partial_profit'
                  ? '%'
                  : '•',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      title: Text(
        '${holding.symbol} (${holding.market.toUpperCase()}) | '
        '${holding.quantity.toStringAsFixed(4)} stuks',
      ),
      subtitle: Text(
        'GAK ${holding.avgEntryPrice.toStringAsFixed(2)} | '
        'nu ${holding.currentPrice.toStringAsFixed(2)} | '
        'doel ${holding.targetPrice.toStringAsFixed(2)} | '
        'stop ${holding.stopLossPrice.toStringAsFixed(2)}\n'
        '${holding.recommendationLabel.toUpperCase()} | score ${holding.positionScore.toStringAsFixed(1)} | ${holding.recommendationReason}',
      ),
      isThreeLine: true,
      trailing: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            '${holding.unrealizedProfitLoss >= 0 ? '+' : ''}${holding.unrealizedProfitLoss.toStringAsFixed(2)}',
            style: TextStyle(color: pnlColor, fontWeight: FontWeight.bold),
          ),
          Text('${holding.unrealizedProfitLossPct.toStringAsFixed(2)}%'),
          TextButton(
            onPressed: () => _sellHolding(holding),
            child: Text(buttonLabel),
          ),
        ],
      ),
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
                Text('Transactiegeschiedenis', style: Theme.of(context).textTheme.titleLarge),
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
              const Text('Nog geen transacties geregistreerd.')
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _history.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) {
                  final trade = _history[_history.length - 1 - i];
                  final isProfit = trade.profitLoss >= 0;
                  final iconColor = trade.action == 'buy'
                      ? Colors.blue
                      : isProfit
                          ? Colors.green
                          : Colors.red;
                  return ListTile(
                    leading: Icon(
                      trade.action == 'buy' ? Icons.add_shopping_cart : Icons.sell,
                      color: iconColor,
                    ),
                    title: Text(
                      '${trade.symbol} ${trade.action.toUpperCase()} | '
                      '${trade.quantity.toStringAsFixed(4)} stuks',
                    ),
                    subtitle: Text(
                      '${trade.timestamp.substring(0, 19).replaceFirst('T', ' ')} | '
                      'prijs ${trade.price.toStringAsFixed(2)} | '
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
