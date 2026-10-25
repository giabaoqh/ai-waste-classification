import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Cho phép truy cập từ điện thoại và thiết bị trong mạng LAN
    port: 5173,
    allowedHosts: true, // Cho phép tất cả các domain tunnel như localtunnel, ngrok, cloudflare
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
})
