/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        glass: '0 20px 80px rgba(7, 14, 32, 0.42)',
      },
      backgroundImage: {
        'hearai-radial': 'radial-gradient(circle at top, rgba(56, 189, 248, 0.18), transparent 42%), radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.12), transparent 30%)',
      },
      keyframes: {
        drift: {
          '0%, 100%': { transform: 'translate3d(0, 0, 0) scale(1)' },
          '50%': { transform: 'translate3d(0, -12px, 0) scale(1.03)' },
        },
        pulseWave: {
          '0%, 100%': { transform: 'scaleY(0.35)', opacity: '0.45' },
          '50%': { transform: 'scaleY(1)', opacity: '1' },
        },
      },
      animation: {
        drift: 'drift 8s ease-in-out infinite',
        pulseWave: 'pulseWave 1.2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};