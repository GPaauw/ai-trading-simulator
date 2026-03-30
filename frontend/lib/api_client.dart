import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models/holding.dart';
import 'models/signal.dart';
import 'models/trade.dart';

// ── Lokale fallback voor development, overschrijf in productie via
// flutter build --dart-define=BACKEND_URL=https://jouw-backend-url
const String kBackendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'https://ai-trading-simulator.onrender.com',
);
// Voor lokale testing kun je tijdelijk hier de token invullen nadat je /login
// hebt aangeroepen. In productie: bewaar tokens in veilige opslag.
String? _token;
// ─────────────────────────────────────────────────────────────────────────────

class ApiClient {
  static Map<String, String> _headers() {
    final headers = {
      'Content-Type': 'application/json',
    };
    if (_token != null) headers['Authorization'] = 'Bearer $_token';
    return headers;
  }

  static void setToken(String token) => _token = token;

  static void clearToken() => _token = null;

  static bool isLoggedIn() => _token != null;

  static Future<List<Signal>> getSignals() async {
    final response = await http.get(Uri.parse('$kBackendUrl/advice'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen signalen (${response.statusCode})');
    }
    final List<dynamic> data = jsonDecode(response.body) as List<dynamic>;
    return data
        .map((e) => Signal.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<List<Holding>> getSellAdvice() async {
    final response = await http.get(Uri.parse('$kBackendUrl/sell-advice'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen verkoopadvies (${response.statusCode})');
    }
    final List<dynamic> data = jsonDecode(response.body) as List<dynamic>;
    return data
        .map((e) => Holding.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<Trade> executeTrade(
    String symbol,
    String action,
    double amount, {
    String? market,
    double? quantity,
    double? price,
    double expectedReturnPct = 0.0,
    double riskPct = 0.0,
  }) async {
    final response = await http.post(
      Uri.parse('$kBackendUrl/trade'),
      headers: _headers(),
      body: jsonEncode({
        'symbol': symbol,
        'action': action,
        'amount': amount,
        'market': market,
        'quantity': quantity,
        'price': price,
        'expected_return_pct': expectedReturnPct,
        'risk_pct': riskPct,
      }),
    );
    if (response.statusCode != 200) {
      throw Exception('Fout bij uitvoeren trade (${response.statusCode})');
    }
    return Trade.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  static Future<List<Trade>> getHistory() async {
    final response = await http.get(Uri.parse('$kBackendUrl/history'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen geschiedenis (${response.statusCode})');
    }
    final List<dynamic> data = jsonDecode(response.body) as List<dynamic>;
    return data
        .map((e) => Trade.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<Map<String, dynamic>> learn() async {
    final response = await http.post(Uri.parse('$kBackendUrl/learn'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij leren (${response.statusCode})');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> getPortfolio() async {
    final response = await http.get(Uri.parse('$kBackendUrl/portfolio'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen portfolio (${response.statusCode})');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<List<Holding>> getHoldings() async {
    final response = await http.get(Uri.parse('$kBackendUrl/holdings'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen holdings (${response.statusCode})');
    }
    final List<dynamic> data = jsonDecode(response.body) as List<dynamic>;
    return data
        .map((e) => Holding.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<Map<String, dynamic>> sendRealtimeAlerts() async {
    final response = await http.post(Uri.parse('$kBackendUrl/alerts/realtime'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception(_extractDetail(response.body, 'Fout bij versturen realtime alerts (${response.statusCode})'));
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> sendDailySummary() async {
    final response = await http.post(Uri.parse('$kBackendUrl/alerts/summary'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception(_extractDetail(response.body, 'Fout bij versturen dag samenvatting (${response.statusCode})'));
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  static String _extractDetail(String body, String fallback) {
    try {
      final map = jsonDecode(body) as Map<String, dynamic>;
      final detail = map['detail'];
      if (detail is String && detail.isNotEmpty) return detail;
    } catch (_) {}
    return fallback;
  }

  static Future<String> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$kBackendUrl/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    );
    if (response.statusCode != 200) {
      throw Exception('Inloggen mislukt (${response.statusCode})');
    }
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final token = data['token'] as String;
    setToken(token);
    return token;
  }
}
