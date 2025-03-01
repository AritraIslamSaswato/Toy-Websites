import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    port: 3000,
    strictPort: false, // Will automatically try next available port
    host: true, // Listen on all addresses
  },
  preview: {
    port: 3000,
    strictPort: false,
    host: true,
  },
  base: '/',
})
