import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models/signal.dart';
import 'models/trade.dart';

// ── Lokale fallback voor development, overschrijf in productie via
// flutter build --dart-define=BACKEND_URL=https://jouw-backend-url
const String kBackendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'https://green-crews-bake.loca.lt',
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

  static Future<List<Signal>> getSignals() async {
    final response = await http.get(Uri.parse('$kBackendUrl/signals'), headers: _headers());
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen signalen (${response.statusCode})');
    }
    final List<dynamic> data = jsonDecode(response.body) as List<dynamic>;
    return data
        .map((e) => Signal.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Future<Trade> executeTrade(
    String symbol,
    String action,
    double amount,
  ) async {
    final response = await http.post(
      Uri.parse('$kBackendUrl/trade'),
      headers: _headers(),
      body: jsonEncode({
        'symbol': symbol,
        'action': action,
        'amount': amount,
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
