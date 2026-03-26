class Holding {
  final String symbol;
  final String market;
  final double quantity;
  final double investedAmount;
  final double avgEntryPrice;
  final double currentPrice;
  final double marketValue;
  final double unrealizedProfitLoss;
  final double unrealizedProfitLossPct;
  final String recommendation;
  final String recommendationReason;
  final double targetPrice;
  final double stopLossPrice;
  final double confidence;
  final double expectedReturnPct;
  final double riskPct;
  final String openedAt;
  final String updatedAt;

  Holding({
    required this.symbol,
    required this.market,
    required this.quantity,
    required this.investedAmount,
    required this.avgEntryPrice,
    required this.currentPrice,
    required this.marketValue,
    required this.unrealizedProfitLoss,
    required this.unrealizedProfitLossPct,
    required this.recommendation,
    required this.recommendationReason,
    required this.targetPrice,
    required this.stopLossPrice,
    required this.confidence,
    required this.expectedReturnPct,
    required this.riskPct,
    required this.openedAt,
    required this.updatedAt,
  });

  factory Holding.fromJson(Map<String, dynamic> json) {
    return Holding(
      symbol: json['symbol'] as String,
      market: json['market'] as String,
      quantity: (json['quantity'] as num).toDouble(),
      investedAmount: (json['invested_amount'] as num).toDouble(),
      avgEntryPrice: (json['avg_entry_price'] as num).toDouble(),
      currentPrice: (json['current_price'] as num).toDouble(),
      marketValue: (json['market_value'] as num).toDouble(),
      unrealizedProfitLoss: (json['unrealized_profit_loss'] as num).toDouble(),
      unrealizedProfitLossPct: (json['unrealized_profit_loss_pct'] as num).toDouble(),
      recommendation: json['recommendation'] as String,
      recommendationReason: json['recommendation_reason'] as String,
      targetPrice: (json['target_price'] as num).toDouble(),
      stopLossPrice: (json['stop_loss_price'] as num).toDouble(),
      confidence: (json['confidence'] as num).toDouble(),
      expectedReturnPct: (json['expected_return_pct'] as num).toDouble(),
      riskPct: (json['risk_pct'] as num).toDouble(),
      openedAt: json['opened_at'] as String,
      updatedAt: json['updated_at'] as String,
    );
  }
}