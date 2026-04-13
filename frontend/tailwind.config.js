/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          50: '#f7f7f8',
          100: '#ededf1',
          200: '#d5d6de',
          300: '#b0b2c0',
          400: '#85889d',
          500: '#666a82',
          600: '#52546b',
          700: '#434557',
          800: '#3a3b4a',
          900: '#1e1f2e',
          950: '#131420',
        },
        accent: {
          green: '#22c55e',
          red: '#ef4444',
          blue: '#3b82f6',
          yellow: '#eab308',
        },
      },
    },
  },
  plugins: [],
}
