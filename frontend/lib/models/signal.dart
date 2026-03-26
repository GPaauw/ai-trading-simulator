class Signal {
  final String id;
  final String symbol;
  final String market;
  final String action;
  final double confidence;
  final double price;
  final double riskPct;
  final double expectedReturnPct;
  final String reason;

  Signal({
    required this.id,
    required this.symbol,
    required this.market,
    required this.action,
    required this.confidence,
    required this.price,
    required this.riskPct,
    required this.expectedReturnPct,
    required this.reason,
  });

  factory Signal.fromJson(Map<String, dynamic> json) {
    return Signal(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      market: (json['market'] as String?) ?? 'unknown',
      action: json['action'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      price: (json['price'] as num).toDouble(),
      riskPct: (json['risk_pct'] as num?)?.toDouble() ?? 0.0,
      expectedReturnPct: (json['expected_return_pct'] as num?)?.toDouble() ?? 0.0,
      reason: (json['reason'] as String?) ?? '',
    );
  }
}
