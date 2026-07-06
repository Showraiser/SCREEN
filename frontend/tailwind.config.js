/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{js,jsx}", "./components/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'DM Serif Display'", "Georgia", "serif"],
        mono:    ["'DM Mono'", "monospace"],
      },
      colors: {
        film: {
          black:  "#0a0a0a",
          gray:   "#1a1a1a",
          border: "#2a2a2a",
          muted:  "#666666",
          accent: "#e8d5a3",
          text:   "#f0ece4",
        },
      },
      animation: {
        "fade-up":    "fadeUp 0.5s ease forwards",
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "scan":       "scan 2s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: 0, transform: "translateY(16px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        scan: {
          "0%":   { backgroundPosition: "0% 0%" },
          "100%": { backgroundPosition: "0% 100%" },
        },
      },
    },
  },
  plugins: [],
};
