/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          // Define your custom colors here
          'wintermute-blue': '#007bff', // Example custom color
          // ... more custom colors
        },
        // Extend other theme properties like fonts, spacing, etc.
        fontFamily: {
          'sans': ['Inter', 'system-ui', 'sans-serif'], // Example custom font
          // ... more custom fonts
        },
      },
    },
    plugins: [
      require("tailwindcss-animate"), // If you are using tailwindcss-animate
      // Add other plugins here if needed
    ],
  }