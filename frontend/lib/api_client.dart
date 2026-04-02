import 'dart:convert';

import 'package:http/http.dart' as http;

// ── Lokale fallback voor development, overschrijf in productie via
// flutter build --dart-define=BACKEND_URL=https://jouw-backend-url
const String kBackendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'https://ai-trading-simulator.onrender.com',
);
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

  static Future<Map<String, dynamic>> getAutoTraderSummary() async {
    final response = await http.get(
      Uri.parse('$kBackendUrl/auto-trader/summary'),
      headers: _headers(),
    );
    if (response.statusCode != 200) {
      throw Exception('Fout bij ophalen auto-trader samenvatting (${response.statusCode})');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}
