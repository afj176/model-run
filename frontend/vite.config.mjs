import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    // `npm run dev` talks to a running Model Run service; its origin check wants no Origin header.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8766",
        changeOrigin: true,
        configure: (proxy) =>
          proxy.on("proxyReq", (req) => req.removeHeader("origin")),
      },
    },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react()],
});
