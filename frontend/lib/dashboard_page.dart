import 'dart:async';

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

class _DashboardPageState extends State<DashboardPage>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  List<Signal> _signals = [];
  List<Signal> _longtermSignals = [];
  List<Holding> _holdings = [];
  List<Trade> _history = [];
  Map<String, dynamic>? _learnResult;
  Map<String, dynamic>? _portfolio;

  bool _loadingSignals = false;
  bool _loadingLongterm = false;
  bool _loadingHoldings = false;
  bool _loadingHistory = false;
  bool _loadingLearn = false;
  bool _loadingPortfolio = false;
  bool _loadingAi = false;

  Map<String, dynamic>? _prefetchStatus;
  Timer? _prefetchPollTimer;

  String? _signalError;
  String? _longtermError;
  String? _aiError;

  List<Map<String, dynamic>> _aiSignals = [];

  // ── Zoek & Track tab state ──
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _searchResults = [];
  List<Map<String, dynamic>> _trackedSymbols = []; // ignore: prefer_final_fields
  List<Map<String, dynamic>> _trackedAnalysis = [];
  bool _loadingSearch = false;
  bool _loadingTrackedAnalysis = false;
  String? _searchError;
  String? _trackedAnalysisError;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 7, vsync: this);
    _refreshAll();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _searchController.dispose();
    _prefetchPollTimer?.cancel();
    super.dispose();
  }

  Future<void> _refreshAll() async {
    _startPrefetchPolling();
    try {
      await Future.wait([
        _refreshSignals(),
        _refreshLongtermSignals(),
        _refreshHoldings(),
        _refreshHistory(),
        _refreshPortfolio(),
        _refreshAiSignals(),
      ]);
    } finally {
      _stopPrefetchPolling();
    }
  }

  void _startPrefetchPolling() {
    _prefetchPollTimer?.cancel();
    _prefetchPollTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      try {
        final status = await ApiClient.getPrefetchStatus();
        if (mounted) setState(() => _prefetchStatus = status);
      } catch (_) {}
    });
  }

  void _stopPrefetchPolling() {
    _prefetchPollTimer?.cancel();
    _prefetchPollTimer = null;
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

  Future<void> _refreshLongtermSignals() async {
    setState(() {
      _loadingLongterm = true;
      _longtermError = null;
    });
    try {
      final signals = await ApiClient.getLongtermSignals();
      if (mounted) setState(() => _longtermSignals = signals);
    } catch (e) {
      if (mounted) setState(() => _longtermError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingLongterm = false);
    }
  }

  Future<void> _refreshAiSignals() async {
    setState(() {
      _loadingAi = true;
      _aiError = null;
    });
    try {
      final signals = await ApiClient.getAiSignals();
      if (mounted) setState(() => _aiSignals = signals);
    } catch (e) {
      if (mounted) setState(() => _aiError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingAi = false);
    }
  }

  Future<void> _registerBuyFromAiSignal(Map<String, dynamic> signal) async {
    // Maak een Signal-object van de AI-data zodat we _registerBuyFromSignal kunnen hergebruiken
    final s = Signal.fromJson({
      ...signal,
      'id': signal['id'] ?? signal['symbol'] ?? '',
    });
    await _registerBuyFromSignal(s);
  }

  Future<void> _refreshHoldings() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingHoldings = true);
    try {
      final holdings = await ApiClient.getSellAdvice();
      if (mounted) setState(() => _holdings = holdings);
    } catch (_) {}
    finally {
      if (mounted) setState(() => _loadingHoldings = false);
    }
  }

  Future<void> _refreshHistory() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingHistory = true);
    try {
      final history = await ApiClient.getHistory();
      if (mounted) setState(() => _history = history);
    } catch (_) {}
    finally {
      if (mounted) setState(() => _loadingHistory = false);
    }
  }

  Future<void> _refreshPortfolio() async {
    if (!ApiClient.isLoggedIn()) return;
    setState(() => _loadingPortfolio = true);
    try {
      final portfolio = await ApiClient.getPortfolio();
      if (mounted) setState(() => _portfolio = portfolio);
    } catch (_) {}
    finally {
      if (mounted) setState(() => _loadingPortfolio = false);
    }
  }

  Future<void> _registerBuyFromSignal(Signal signal) async {
    if (!ApiClient.isLoggedIn()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Log eerst in om een aankoop te registreren.')),
      );
      return;
    }

    final currentBalance =
        (_portfolio?['available_cash'] as num?)?.toDouble() ?? 2000.0;
    final suggestedAmount =
        (currentBalance * (signal.riskPct / 100.0) * 8.0)
            .clamp(100.0, currentBalance * 0.4);
    final amountController =
        TextEditingController(text: suggestedAmount.toStringAsFixed(2));
    final priceController =
        TextEditingController(text: signal.price.toStringAsFixed(4));
    final quantityController = TextEditingController();

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Aankoop registreren: ${signal.symbol}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: amountController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Totaal gekocht bedrag (€)'),
            ),
            TextField(
              controller: priceController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Aankoopprijs per stuk'),
            ),
            TextField(
              controller: quantityController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Aantal stuks (optioneel)'),
            ),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: const Text('Annuleren')),
          FilledButton(
              onPressed: () => Navigator.of(ctx).pop(true),
              child: const Text('Opslaan')),
        ],
      ),
    );

    if (confirmed != true) return;

    final amount =
        double.tryParse(amountController.text.replaceAll(',', '.'));
    final price =
        double.tryParse(priceController.text.replaceAll(',', '.'));
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
      await Future.wait([
        _refreshHistory(),
        _refreshPortfolio(),
        _refreshHoldings(),
      ]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Tradefout: $e'),
            backgroundColor: Colors.red[700]),
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
        setState(
            () => _learnResult = result['parameters'] as Map<String, dynamic>?);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Leerfout: $e'),
            backgroundColor: Colors.red[700]),
      );
    } finally {
      if (mounted) setState(() => _loadingLearn = false);
    }
  }

  Future<void> _sellAll() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Alle posities verkopen'),
        content: const Text(
          'Einde dag: wil je alle open posities sluiten?\n'
          'Dit verkoopt alles tegen de huidige marktprijs.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: const Text('Annuleren')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.of(ctx).pop(true),
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
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              '$sold posities verkocht | Totaal P/L: €${pl.toStringAsFixed(2)}'),
          backgroundColor: pl >= 0 ? Colors.green[700] : Colors.red[700],
        ),
      );
      await Future.wait([
        _refreshHistory(),
        _refreshPortfolio(),
        _refreshHoldings(),
      ]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Fout bij verkopen: $e'),
            backgroundColor: Colors.red[700]),
      );
    }
  }

  Future<void> _sellHolding(Holding holding) async {
    final defaultQuantity = holding.suggestedSellFraction > 0
        ? holding.quantity * holding.suggestedSellFraction
        : holding.quantity;
    final quantityController =
        TextEditingController(text: defaultQuantity.toStringAsFixed(6));
    final priceController =
        TextEditingController(text: holding.currentPrice.toStringAsFixed(4));

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Verkoop ${holding.symbol}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: quantityController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Aantal te verkopen'),
            ),
            TextField(
              controller: priceController,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(
                  labelText: 'Verkoopprijs per stuk'),
            ),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: const Text('Annuleren')),
          FilledButton(
              onPressed: () => Navigator.of(ctx).pop(true),
              child: const Text('Verkoop')),
        ],
      ),
    );

    if (confirmed != true) return;

    final quantity =
        double.tryParse(quantityController.text.replaceAll(',', '.'));
    final price =
        double.tryParse(priceController.text.replaceAll(',', '.'));
    if (quantity == null ||
        quantity <= 0 ||
        price == null ||
        price <= 0) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Vul een geldige hoeveelheid en prijs in.')),
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
      await Future.wait([
        _refreshHistory(),
        _refreshPortfolio(),
        _refreshHoldings(),
      ]);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Verkoopfout: $e'),
            backgroundColor: Colors.red[700]),
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
            backgroundColor: Colors.red[700]),
      );
    }
  }

  Future<void> _sendDailySummary() async {
    try {
      final result = await ApiClient.sendDailySummary();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Dagelijkse samenvatting: ${result.toString()}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Summaryfout: $e'),
            backgroundColor: Colors.red[700]),
      );
    }
  }

  // ════════════════════════════════════════════════════════════════════
  // LOADING INDICATOR
  // ════════════════════════════════════════════════════════════════════

  Widget _buildLoadingIndicator() {
    final status = _prefetchStatus;
    final isRunning = status?['is_running'] as bool? ?? false;
    final pct = (status?['progress_pct'] as num?)?.toDouble();
    final eta = (status?['eta_seconds'] as num?)?.toInt();

    if (isRunning && pct != null) {
      final etaText = (eta != null && eta > 0) ? ' — nog ~${eta}s' : '';
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          LinearProgressIndicator(value: pct / 100),
          const SizedBox(height: 8),
          Text(
            'Marktdata laden... ${pct.toStringAsFixed(0)}%$etaText',
            style: const TextStyle(fontSize: 12),
          ),
        ],
      );
    }
    return const Center(child: CircularProgressIndicator());
  }

  // ════════════════════════════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════════════════════════════

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
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true,
          tabs: const [
            Tab(icon: Icon(Icons.auto_awesome), text: 'AI Analyse'),
            Tab(icon: Icon(Icons.show_chart), text: 'Aandelen'),
            Tab(icon: Icon(Icons.currency_bitcoin), text: 'Crypto'),
            Tab(icon: Icon(Icons.diamond), text: 'Grondstoffen'),
            Tab(icon: Icon(Icons.trending_up), text: 'Langetermijn'),
            Tab(icon: Icon(Icons.search), text: 'Zoeken'),
            Tab(icon: Icon(Icons.account_balance_wallet), text: 'Portfolio'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildAiTab(),
          _buildMarketTab('us', 'Aandelen'),
          _buildMarketTab('crypto', 'Crypto'),
          _buildMarketTab('commodity', 'Grondstoffen'),
          _buildLongtermTab(),
          _buildSearchTab(),
          _buildPortfolioTab(),
        ],
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // AI ANALYSE TAB
  // ════════════════════════════════════════════════════════════════════

  Widget _buildAiTab() {
    return RefreshIndicator(
      onRefresh: _refreshAiSignals,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 980),
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text('AI Analyse — Top signalen',
                              style: Theme.of(context).textTheme.titleLarge),
                        ),
                        IconButton(
                          icon: const Icon(Icons.refresh),
                          onPressed: _loadingAi ? null : _refreshAiSignals,
                        ),
                      ],
                    ),
                    const Divider(),
                    if (_loadingAi)
                      Padding(
                        padding: const EdgeInsets.all(12),
                        child: _buildLoadingIndicator(),
                      )
                    else if (_aiError != null)
                      Text(_aiError!,
                          style: TextStyle(
                              color: Theme.of(context).colorScheme.error))
                    else if (_aiSignals.isEmpty)
                      const Text(
                          'Geen signalen beschikbaar. Data wordt geladen...')
                    else
                      ListView.separated(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: _aiSignals.length,
                        separatorBuilder: (_, __) => const Divider(height: 1),
                        itemBuilder: (_, i) =>
                            _buildAiSignalTile(_aiSignals[i]),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAiSignalTile(Map<String, dynamic> signal) {
    final symbol = signal['symbol'] as String? ?? '';
    final market = signal['market'] as String? ?? '';
    final price = (signal['price'] as num?)?.toDouble() ?? 0.0;
    final aiScore = (signal['ai_score'] as num?)?.toInt() ?? 0;
    final aiAnalysis = signal['ai_analysis'] as String? ?? '';
    final aiRisk = signal['ai_risk'] as String? ?? '';
    final action = signal['action'] as String? ?? 'buy';
    final confidence = (signal['confidence'] as num?)?.toDouble() ?? 0.0;
    final expectedReturn =
        (signal['expected_return_pct'] as num?)?.toDouble() ?? 0.0;
    final targetPrice =
        (signal['target_price'] as num?)?.toDouble() ?? 0.0;
    final rankLabel = signal['rank_label'] as String? ?? '';

    final scoreColor = aiScore >= 75
        ? Colors.green
        : aiScore >= 50
            ? Colors.orange
            : Colors.red;
    final actionColor = action == 'buy' ? Colors.green : (action == 'sell' ? Colors.red : Colors.grey);
    final loggedIn = ApiClient.isLoggedIn();

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: scoreColor,
        child: Text(
          '$aiScore',
          style: const TextStyle(
              color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13),
        ),
      ),
      title: Text(
        '$symbol (${_marketLabel(market)}) — ${action.toUpperCase()} @ ${price.toStringAsFixed(2)}',
      ),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 4),
          if (aiAnalysis.isNotEmpty)
            Text(
              aiAnalysis,
              style: const TextStyle(fontStyle: FontStyle.italic),
            ),
          if (aiRisk.isNotEmpty) ...[
            const SizedBox(height: 2),
            Text(
              'Risico: $aiRisk',
              style: TextStyle(color: Colors.red[300], fontSize: 12),
            ),
          ],
          const SizedBox(height: 4),
          Text(
            '${rankLabel.isNotEmpty ? '$rankLabel | ' : ''}'
            'Doel: €${targetPrice.toStringAsFixed(2)} | '
            '+${expectedReturn.toStringAsFixed(2)}% | '
            'Zekerheid: ${(confidence * 100).toStringAsFixed(0)}%',
            style: const TextStyle(fontSize: 12),
          ),
        ],
      ),
      isThreeLine: true,
      trailing: loggedIn && action == 'buy'
          ? FilledButton(
              style: FilledButton.styleFrom(backgroundColor: actionColor),
              onPressed: () => _registerBuyFromAiSignal(signal),
              child: const Text('Ik kocht dit'),
            )
          : const SizedBox.shrink(),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // ZOEK & TRACK TAB
  // ════════════════════════════════════════════════════════════════════

  Future<void> _searchSymbol(String query) async {
    if (query.trim().isEmpty) return;
    setState(() {
      _loadingSearch = true;
      _searchError = null;
    });
    try {
      final results = await ApiClient.searchSymbols(query.trim());
      if (mounted) setState(() => _searchResults = results);
    } catch (e) {
      if (mounted) setState(() => _searchError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingSearch = false);
    }
  }

  void _addToTracked(Map<String, dynamic> result) {
    final symbol = (result['symbol'] as String).toUpperCase();
    final alreadyTracked =
        _trackedSymbols.any((t) => (t['symbol'] as String).toUpperCase() == symbol);
    if (alreadyTracked) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('$symbol staat al in je lijst.')),
      );
      return;
    }
    setState(() {
      _trackedSymbols.add({
        'symbol': symbol,
        'market': result['market'] ?? 'us',
        'buy_price': 0.0,
        'quantity': 0.0,
      });
    });
  }

  void _removeFromTracked(int index) {
    setState(() {
      _trackedSymbols.removeAt(index);
      // Verwijder bijbehorende analyse
      if (_trackedAnalysis.isNotEmpty) {
        _trackedAnalysis = _trackedAnalysis
            .where((a) => _trackedSymbols.any(
                (t) => (t['symbol'] as String).toUpperCase() == (a['symbol'] as String? ?? '').toUpperCase()))
            .toList();
      }
    });
  }

  Future<void> _editTrackedEntry(int index) async {
    final entry = _trackedSymbols[index];
    final priceController =
        TextEditingController(text: (entry['buy_price'] as num).toStringAsFixed(2));
    final quantityController =
        TextEditingController(text: (entry['quantity'] as num).toStringAsFixed(4));

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('${entry['symbol']} — Aankoopgegevens'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: priceController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Aankoopprijs per stuk (€)'),
            ),
            TextField(
              controller: quantityController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: const InputDecoration(labelText: 'Aantal stuks'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Annuleren'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Opslaan'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    final price = double.tryParse(priceController.text.replaceAll(',', '.')) ?? 0.0;
    final quantity = double.tryParse(quantityController.text.replaceAll(',', '.')) ?? 0.0;

    setState(() {
      _trackedSymbols[index] = {
        ...entry,
        'buy_price': price,
        'quantity': quantity,
      };
    });
  }

  Future<void> _analyzeTrackedSymbols() async {
    if (_trackedSymbols.isEmpty) return;
    if (!ApiClient.isLoggedIn()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Log eerst in voor AI-analyse.')),
      );
      return;
    }

    setState(() {
      _loadingTrackedAnalysis = true;
      _trackedAnalysisError = null;
    });

    try {
      final results = await ApiClient.analyzeCustomSymbols(_trackedSymbols);
      if (mounted) setState(() => _trackedAnalysis = results);
    } catch (e) {
      if (mounted) setState(() => _trackedAnalysisError = e.toString());
    } finally {
      if (mounted) setState(() => _loadingTrackedAnalysis = false);
    }
  }

  Future<void> _registerBuyFromTracked(Map<String, dynamic> entry) async {
    final symbol = entry['symbol'] as String;
    final market = entry['market'] as String? ?? 'us';
    final buyPrice = (entry['buy_price'] as num?)?.toDouble() ?? 0.0;
    final quantity = (entry['quantity'] as num?)?.toDouble() ?? 0.0;

    if (buyPrice <= 0 || quantity <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Vul eerst aankoopprijs en aantal in.')),
      );
      return;
    }

    final amount = buyPrice * quantity;
    try {
      final trade = await ApiClient.executeTrade(
        symbol,
        'buy',
        amount,
        market: market,
        quantity: quantity,
        price: buyPrice,
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
        SnackBar(content: Text('Tradefout: $e'), backgroundColor: Colors.red[700]),
      );
    }
  }

  Widget _buildSearchTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 980),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── Zoekbalk ──
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Aandeel zoeken',
                          style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _searchController,
                              decoration: const InputDecoration(
                                hintText: 'Ticker/symbool (bijv. AAPL, TSLA, BTC)',
                                prefixIcon: Icon(Icons.search),
                                border: OutlineInputBorder(),
                              ),
                              textCapitalization: TextCapitalization.characters,
                              onSubmitted: _searchSymbol,
                            ),
                          ),
                          const SizedBox(width: 8),
                          FilledButton.icon(
                            icon: _loadingSearch
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2, color: Colors.white),
                                  )
                                : const Icon(Icons.search),
                            label: const Text('Zoek'),
                            onPressed: _loadingSearch
                                ? null
                                : () => _searchSymbol(_searchController.text),
                          ),
                        ],
                      ),
                      if (_searchError != null) ...[
                        const SizedBox(height: 8),
                        Text(_searchError!,
                            style: TextStyle(
                                color: Theme.of(context).colorScheme.error)),
                      ],
                      if (_searchResults.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        const Divider(),
                        ListView.separated(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: _searchResults.length,
                          separatorBuilder: (_, __) => const Divider(height: 1),
                          itemBuilder: (_, i) {
                            final r = _searchResults[i];
                            final symbol = r['symbol'] as String? ?? '';
                            final market = r['market'] as String? ?? 'us';
                            final price = (r['price'] as num?)?.toDouble();
                            final inWatchlist = r['in_watchlist'] as bool? ?? false;
                            return ListTile(
                              leading: CircleAvatar(
                                backgroundColor:
                                    inWatchlist ? Colors.green : Colors.blueGrey,
                                child: Text(
                                  _marketLabel(market).substring(0, 2),
                                  style: const TextStyle(
                                      color: Colors.white, fontSize: 12),
                                ),
                              ),
                              title: Text('$symbol (${_marketLabel(market)})'),
                              subtitle: Text(price != null
                                  ? 'Prijs: €${price.toStringAsFixed(2)}'
                                  : inWatchlist
                                      ? 'In watchlist'
                                      : ''),
                              trailing: IconButton(
                                icon: const Icon(Icons.add_circle_outline),
                                tooltip: 'Toevoegen aan lijst',
                                onPressed: () => _addToTracked(r),
                              ),
                            );
                          },
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // ── Gevolgde symbolen ──
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text('Mijn aandelen',
                              style: Theme.of(context).textTheme.titleLarge),
                          Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              if (_trackedSymbols.isNotEmpty)
                                FilledButton.icon(
                                  icon: _loadingTrackedAnalysis
                                      ? const SizedBox(
                                          width: 16,
                                          height: 16,
                                          child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                              color: Colors.white),
                                        )
                                      : const Icon(Icons.auto_awesome),
                                  label: const Text('AI Analyse'),
                                  onPressed: _loadingTrackedAnalysis
                                      ? null
                                      : _analyzeTrackedSymbols,
                                ),
                            ],
                          ),
                        ],
                      ),
                      const Divider(),
                      if (_trackedSymbols.isEmpty)
                        const Padding(
                          padding: EdgeInsets.all(12),
                          child: Text(
                            'Zoek hierboven een aandeel en voeg het toe aan je lijst. '
                            'Vul je aankoopprijs en aantal in, en laat AI het analyseren.',
                          ),
                        )
                      else
                        ListView.separated(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: _trackedSymbols.length,
                          separatorBuilder: (_, __) => const Divider(height: 1),
                          itemBuilder: (_, i) {
                            final entry = _trackedSymbols[i];
                            final symbol = entry['symbol'] as String;
                            final market = entry['market'] as String? ?? 'us';
                            final buyPrice =
                                (entry['buy_price'] as num?)?.toDouble() ?? 0.0;
                            final quantity =
                                (entry['quantity'] as num?)?.toDouble() ?? 0.0;
                            final loggedIn = ApiClient.isLoggedIn();

                            return ListTile(
                              leading: CircleAvatar(
                                backgroundColor: Colors.indigo,
                                child: Text(
                                  symbol.length >= 2
                                      ? symbol.substring(0, 2)
                                      : symbol,
                                  style: const TextStyle(
                                      color: Colors.white, fontSize: 12),
                                ),
                              ),
                              title: Text(
                                  '$symbol (${_marketLabel(market)})'),
                              subtitle: Text(
                                buyPrice > 0
                                    ? 'Aankoop: €${buyPrice.toStringAsFixed(2)} | '
                                        '${quantity.toStringAsFixed(4)} stuks | '
                                        'Totaal: €${(buyPrice * quantity).toStringAsFixed(2)}'
                                    : 'Tik op bewerken om aankoopgegevens in te vullen',
                              ),
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  IconButton(
                                    icon: const Icon(Icons.edit, size: 20),
                                    tooltip: 'Bewerken',
                                    onPressed: () => _editTrackedEntry(i),
                                  ),
                                  if (loggedIn && buyPrice > 0 && quantity > 0)
                                    IconButton(
                                      icon: const Icon(Icons.add_shopping_cart,
                                          size: 20),
                                      tooltip: 'Registreer als aankoop',
                                      onPressed: () =>
                                          _registerBuyFromTracked(entry),
                                    ),
                                  IconButton(
                                    icon: const Icon(Icons.delete_outline,
                                        size: 20, color: Colors.red),
                                    tooltip: 'Verwijderen',
                                    onPressed: () => _removeFromTracked(i),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
                    ],
                  ),
                ),
              ),

              // ── AI Analyse resultaten ──
              if (_trackedAnalysis.isNotEmpty || _loadingTrackedAnalysis || _trackedAnalysisError != null) ...[
                const SizedBox(height: 16),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('AI Analyse — Mijn aandelen',
                            style: Theme.of(context).textTheme.titleLarge),
                        const Divider(),
                        if (_loadingTrackedAnalysis)
                          const Padding(
                            padding: EdgeInsets.all(12),
                            child: Center(child: CircularProgressIndicator()),
                          )
                        else if (_trackedAnalysisError != null)
                          Text(_trackedAnalysisError!,
                              style: TextStyle(
                                  color: Theme.of(context).colorScheme.error))
                        else
                          ListView.separated(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            itemCount: _trackedAnalysis.length,
                            separatorBuilder: (_, __) =>
                                const Divider(height: 1),
                            itemBuilder: (_, i) =>
                                _buildAiSignalTile(_trackedAnalysis[i]),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // MARKET TAB (Aandelen / Crypto / Grondstoffen)
  // ════════════════════════════════════════════════════════════════════

  Widget _buildMarketTab(String market, String label) {
    final marketSignals =
        _signals.where((s) => s.market == market).toList();
    final marketHoldings =
        _holdings.where((h) => h.market == market).toList();

    return RefreshIndicator(
      onRefresh: () async {
        await Future.wait([_refreshSignals(), _refreshHoldings()]);
      },
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 980),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // -- Signalen --
                _buildSignalsCard(
                  title: 'Daytrades — $label',
                  signals: marketSignals,
                  loading: _loadingSignals,
                  error: _signalError,
                  onRefresh: _refreshSignals,
                ),
                const SizedBox(height: 16),
                // -- Posities --
                _buildHoldingsCard(
                  title: 'Posities — $label',
                  holdings: marketHoldings,
                  loading: _loadingHoldings,
                  onRefresh: _refreshHoldings,
                  showSellAll: marketHoldings.isNotEmpty,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // LANGETERMIJN TAB
  // ════════════════════════════════════════════════════════════════════

  Widget _buildLongtermTab() {
    return RefreshIndicator(
      onRefresh: _refreshLongtermSignals,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 980),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _buildSignalsCard(
                  title: 'Langetermijn adviezen — alle markten',
                  signals: _longtermSignals,
                  loading: _loadingLongterm,
                  error: _longtermError,
                  onRefresh: _refreshLongtermSignals,
                  isLongterm: true,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // PORTFOLIO TAB
  // ════════════════════════════════════════════════════════════════════

  Widget _buildPortfolioTab() {
    return RefreshIndicator(
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
                const SizedBox(height: 16),
                _buildAllHoldingsCard(),
                const SizedBox(height: 16),
                _buildActionsCard(),
                const SizedBox(height: 16),
                _buildHistoryCard(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // SHARED WIDGETS
  // ════════════════════════════════════════════════════════════════════

  Widget _buildSignalsCard({
    required String title,
    required List<Signal> signals,
    required bool loading,
    String? error,
    required VoidCallback onRefresh,
    bool isLongterm = false,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(title,
                      style: Theme.of(context).textTheme.titleLarge),
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: loading ? null : onRefresh,
                ),
              ],
            ),
            const Divider(),
            if (loading)
              Padding(
                padding: const EdgeInsets.all(12),
                child: _buildLoadingIndicator(),
              )
            else if (error != null)
              Text(error,
                  style:
                      TextStyle(color: Theme.of(context).colorScheme.error))
            else if (signals.isEmpty)
              Text(isLongterm
                  ? 'Geen langetermijn koopkansen gevonden.'
                  : 'Geen daytrade-kansen op dit moment.')
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: signals.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) =>
                    _buildSignalTile(signals[i], isLongterm: isLongterm),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildSignalTile(Signal signal, {bool isLongterm = false}) {
    final loggedIn = ApiClient.isLoggedIn();
    final actionColor = signal.action == 'buy' ? Colors.green : (signal.action == 'sell' ? Colors.red : Colors.grey);
    final horizonText = isLongterm
        ? '~${signal.expectedDays} dagen'
        : 'vandaag (daytrade)';
    final marketLabel = _marketLabel(signal.market);
    final hasAi = signal.aiScore > 0;

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: hasAi
            ? (signal.aiScore >= 75 ? Colors.green : (signal.aiScore >= 50 ? Colors.orange : Colors.red))
            : actionColor,
        child: hasAi
            ? Text('${signal.aiScore}',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13))
            : Text(signal.action == 'buy' ? '↑' : '↓',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ),
      title: Text(
        '${signal.symbol} ($marketLabel)  —  '
        '${signal.action.toUpperCase()} @ ${signal.price.toStringAsFixed(2)}',
      ),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (hasAi && signal.aiAnalysis.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(signal.aiAnalysis, style: const TextStyle(fontStyle: FontStyle.italic)),
          ],
          if (hasAi && signal.aiRisk.isNotEmpty) ...[
            const SizedBox(height: 2),
            Text('Risico: ${signal.aiRisk}',
                style: TextStyle(color: Colors.red[300], fontSize: 12)),
          ],
          Text(
            '${signal.rankLabel.isEmpty ? (isLongterm ? 'Langetermijn' : 'Daytrade') : signal.rankLabel} | '
            'score ${signal.rankingScore.toStringAsFixed(1)}\n'
            'Verwacht: +${signal.expectedReturnPct.toStringAsFixed(2)}% → '
            'doel €${signal.targetPrice.toStringAsFixed(2)} | '
            'winst ~€${signal.expectedProfit.toStringAsFixed(0)} per €1000 | '
            '$horizonText\n'
            'Stop-loss: ${signal.riskPct.toStringAsFixed(2)}% | '
            'Zekerheid: ${(signal.confidence * 100).toStringAsFixed(0)}%',
          ),
        ],
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

  Widget _buildHoldingsCard({
    required String title,
    required List<Holding> holdings,
    required bool loading,
    required VoidCallback onRefresh,
    bool showSellAll = false,
  }) {
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
                Expanded(
                  child: Text(title,
                      style: Theme.of(context).textTheme.titleLarge),
                ),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (showSellAll)
                      FilledButton.icon(
                        style: FilledButton.styleFrom(
                            backgroundColor: Colors.red),
                        icon: const Icon(Icons.sell, size: 18),
                        label: const Text('Verkoop alles'),
                        onPressed: _sellAll,
                      ),
                    const SizedBox(width: 8),
                    IconButton(
                      icon: const Icon(Icons.refresh),
                      onPressed: loading ? null : onRefresh,
                    ),
                  ],
                ),
              ],
            ),
            const Divider(),
            if (!loggedIn)
              const Text(
                  'Log in om verkoopadvies voor je bezittingen te bekijken.')
            else if (loading)
              const Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(),
              )
            else if (holdings.isEmpty)
              const Text('Geen posities in deze categorie.')
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: holdings.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) => _buildHoldingTile(holdings[i]),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHoldingTile(Holding holding) {
    final pnlColor =
        holding.unrealizedProfitLoss >= 0 ? Colors.green : Colors.red;
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
          style: const TextStyle(
              color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      title: Text(
        '${holding.symbol} (${_marketLabel(holding.market)}) | '
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
            style:
                TextStyle(color: pnlColor, fontWeight: FontWeight.bold),
          ),
          Text(
              '${holding.unrealizedProfitLossPct.toStringAsFixed(2)}%'),
          TextButton(
            onPressed: () => _sellHolding(holding),
            child: Text(buttonLabel),
          ),
        ],
      ),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // PORTFOLIO TAB WIDGETS
  // ════════════════════════════════════════════════════════════════════

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
                Text('Portfolio overzicht',
                    style: Theme.of(context).textTheme.titleLarge),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _loadingPortfolio ? null : _refreshPortfolio,
                ),
              ],
            ),
            const Divider(),
            if (!loggedIn)
              const Text('Log in om je portfolio te bekijken.')
            else if (_loadingPortfolio)
              const Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(),
              )
            else if (_portfolio == null)
              const Text('Portfolio nog niet geladen.')
            else ...[
              Wrap(
                spacing: 20,
                runSpacing: 8,
                children: [
                  Text(
                      'Start: €${(_portfolio!['start_balance'] as num).toStringAsFixed(2)}'),
                  Text(
                      'Nu: €${(_portfolio!['current_balance'] as num).toStringAsFixed(2)}'),
                  Text(
                      'Cash: €${(_portfolio!['available_cash'] as num).toStringAsFixed(2)}'),
                  Text(
                      'Marktwaarde: €${(_portfolio!['market_value'] as num).toStringAsFixed(2)}'),
                  Text(
                      'Totaal P/L: €${(_portfolio!['total_profit_loss'] as num).toStringAsFixed(2)}'),
                  Text('Posities: ${_portfolio!['holdings_count']}'),
                  Text(
                      'Trades vandaag: ${_portfolio!['daily_trade_count']} / 50'),
                ],
              ),
              const SizedBox(height: 12),
              _buildDailyPnlRow(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDailyPnlRow() {
    final realized =
        (_portfolio?['daily_realized_pnl'] as num?)?.toDouble() ?? 0.0;
    final unrealized =
        (_portfolio?['daily_unrealized_pnl'] as num?)?.toDouble() ?? 0.0;
    final total =
        (_portfolio?['daily_total_pnl'] as num?)?.toDouble() ?? 0.0;
    final totalColor = total >= 0 ? Colors.green : Colors.red;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: totalColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Dagelijkse P/L',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          Wrap(
            spacing: 20,
            runSpacing: 4,
            children: [
              Text('Gerealiseerd: €${realized.toStringAsFixed(2)}'),
              Text('Ongerealiseerd: €${unrealized.toStringAsFixed(2)}'),
              Text(
                'Totaal: €${total.toStringAsFixed(2)}',
                style: TextStyle(
                    color: totalColor, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAllHoldingsCard() {
    return _buildHoldingsCard(
      title: 'Alle posities',
      holdings: _holdings,
      loading: _loadingHoldings,
      onRefresh: _refreshHoldings,
      showSellAll: _holdings.isNotEmpty,
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
            Text('Leren & Alerts',
                style: Theme.of(context).textTheme.titleLarge),
            const Divider(),
            if (!loggedIn)
              const Text(
                  'Log in om leren en e-mailalerts te gebruiken.')
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
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.psychology),
                    label: const Text('Leer van markt + trades'),
                    onPressed: _loadingLearn ? null : _learn,
                  ),
                  OutlinedButton.icon(
                    icon: const Icon(
                        Icons.notifications_active_outlined),
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
                Text('Transactiegeschiedenis',
                    style: Theme.of(context).textTheme.titleLarge),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed:
                      _loadingHistory ? null : _refreshHistory,
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
                      trade.action == 'buy'
                          ? Icons.add_shopping_cart
                          : Icons.sell,
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

  // ════════════════════════════════════════════════════════════════════
  // HELPERS
  // ════════════════════════════════════════════════════════════════════

  String _marketLabel(String market) {
    switch (market) {
      case 'us':
        return 'US';
      case 'crypto':
        return 'CRYPTO';
      case 'commodity':
        return 'GRONDSTOF';
      default:
        return market.toUpperCase();
    }
  }
}
