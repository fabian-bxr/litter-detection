import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    proxy: {
      '/api': 'http://localhost:8090',
      '/ws': { target: 'ws://localhost:8090', ws: true },
    },
  },
})
