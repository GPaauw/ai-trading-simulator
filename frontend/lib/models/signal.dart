class Signal {
  final String id;
  final String symbol;
  final String action;
  final double confidence;
  final double price;

  Signal({
    required this.id,
    required this.symbol,
    required this.action,
    required this.confidence,
    required this.price,
  });

  factory Signal.fromJson(Map<String, dynamic> json) {
    return Signal(
      id: json['id'] as String,
      symbol: json['symbol'] as String,
      action: json['action'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      price: (json['price'] as num).toDouble(),
    );
  }
}
