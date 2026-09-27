import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, /api is served by the FastAPI app: uvicorn airq.api:app --port 8000
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5190,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
