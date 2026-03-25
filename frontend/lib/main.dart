import 'package:flutter/material.dart';

import 'login_page.dart';

void main() {
  runApp(const TradingSimulatorApp());
}

class TradingSimulatorApp extends StatelessWidget {
  const TradingSimulatorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Trading Simulator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const LoginPage(),
    );
  }
}
