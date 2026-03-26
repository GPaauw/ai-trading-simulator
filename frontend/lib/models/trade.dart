class Trade {
  final String id;
  final String symbol;
  final String action;
  final double amount;
  final double profitLoss;
  final String timestamp;
  final String status;
  final double expectedReturnPct;
  final double riskPct;

  Trade({
    required this.id,
    required this.symbol,
    required this.action,
    required this.amount,
    required this.profitLoss,
    required this.timestamp,
    required this.status,
    this.expectedReturnPct = 0.0,
    this.riskPct = 0.0,
  });

  factory Trade.fromJson(Map<String, dynamic> json) {
    return Trade(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      action: json['action'] as String,
      amount: (json['amount'] as num).toDouble(),
      profitLoss: (json['profit_loss'] as num).toDouble(),
      timestamp: json['timestamp'] as String,
      status: json['status'] as String,
      expectedReturnPct: (json['expected_return_pct'] as num?)?.toDouble() ?? 0.0,
      riskPct: (json['risk_pct'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
