import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Khi chạy dev, mọi request /api/* được chuyển tiếp sang backend (mặc định http://localhost:8000)
// => frontend gọi "/api/ask" mà không cần lo CORS.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": { target: env.BACKEND_URL || "http://localhost:8000", changeOrigin: true },
      },
    },
  };
});
