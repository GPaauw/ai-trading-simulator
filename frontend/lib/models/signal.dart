class Signal {
  final String id;
  final String symbol;
  final String market;
  final String action;
  final double confidence;
  final double price;
  final double riskPct;
  final double expectedReturnPct;
  final double targetPrice;
  final double expectedProfit;
  final int expectedDays;
  final double rankingScore;
  final String rankLabel;
  final String reason;
  final String strategy;
  // AI-analyse velden
  final int aiScore;
  final String aiAnalysis;
  final String aiRisk;

  Signal({
    required this.id,
    required this.symbol,
    required this.market,
    required this.action,
    required this.confidence,
    required this.price,
    required this.riskPct,
    required this.expectedReturnPct,
    required this.targetPrice,
    required this.expectedProfit,
    required this.expectedDays,
    required this.rankingScore,
    required this.rankLabel,
    required this.reason,
    required this.strategy,
    this.aiScore = 0,
    this.aiAnalysis = '',
    this.aiRisk = '',
  });

  factory Signal.fromJson(Map<String, dynamic> json) {
    return Signal(
      id: (json['id'] as String?) ?? '',
      symbol: json['symbol'] as String,
      market: (json['market'] as String?) ?? 'unknown',
      action: json['action'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      price: (json['price'] as num).toDouble(),
      riskPct: (json['risk_pct'] as num?)?.toDouble() ?? 0.0,
      expectedReturnPct: (json['expected_return_pct'] as num?)?.toDouble() ?? 0.0,
      targetPrice: (json['target_price'] as num?)?.toDouble() ?? 0.0,
      expectedProfit: (json['expected_profit'] as num?)?.toDouble() ?? 0.0,
      expectedDays: (json['expected_days'] as int?) ?? 0,
      rankingScore: (json['ranking_score'] as num?)?.toDouble() ?? 0.0,
      rankLabel: (json['rank_label'] as String?) ?? '',
      reason: (json['reason'] as String?) ?? '',
      strategy: (json['strategy'] as String?) ?? 'daytrade',
      aiScore: (json['ai_score'] as num?)?.toInt() ?? 0,
      aiAnalysis: (json['ai_analysis'] as String?) ?? '',
      aiRisk: (json['ai_risk'] as String?) ?? '',
    );
  }
}
