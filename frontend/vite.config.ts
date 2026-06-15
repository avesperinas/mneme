import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on :5173; the API base URL is read at runtime from VITE_API_BASE_URL.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
