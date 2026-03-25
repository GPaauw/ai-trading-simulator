class Trade {
  final String id;
  final String symbol;
  final String action;
  final double amount;
  final double profitLoss;
  final String timestamp;
  final String status;

  Trade({
    required this.id,
    required this.symbol,
    required this.action,
    required this.amount,
    required this.profitLoss,
    required this.timestamp,
    required this.status,
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
    );
  }
}
