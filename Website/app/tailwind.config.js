/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        'annotation-green': {
          DEFAULT: '#d9ead3',
        },
        'annotation-red': {
          DEFAULT: '#f4cccc',
        },
        'neon-yellow': {
          DEFAULT: '#FFF01F',
        },
      },
    },
  },
  plugins: [],
}
