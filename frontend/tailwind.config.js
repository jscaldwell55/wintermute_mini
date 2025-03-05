// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Correctly scans your source files
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('tailwindcss-animate') // Correctly requires the plugin
  ],
}