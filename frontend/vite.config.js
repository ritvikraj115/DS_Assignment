import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/react/') || id.includes('/node_modules/react-dom/')) return 'react-vendor'
          if (id.includes('/node_modules/lucide-react/')) return 'icons'
          if (id.includes('/node_modules/react-markdown/') || id.includes('/node_modules/rehype-') || id.includes('/node_modules/remark-') || id.includes('/node_modules/highlight.js/')) return 'markdown'
        },
      },
    },
  },
})
