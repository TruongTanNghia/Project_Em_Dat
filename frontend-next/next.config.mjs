/**
 * BACKEND_URL is where /api/* requests get proxied to. Set this in:
 *   • .env.local           (for `npm run dev` on your machine)
 *   • Vercel project env   (for production — paste your ngrok HTTPS URL)
 * Falls back to localhost:5000 if not set.
 *
 * NO trailing slash.
 */
const BACKEND_URL = (process.env.BACKEND_URL || 'http://localhost:5000').replace(/\/+$/, '');

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable image optimization (we serve raw GLB / PNG / base64 images
  // directly — no Next/Image usage in the legacy app).
  images: { unoptimized: true },

  // Server-side proxy: every /api/* request from the legacy app gets
  // forwarded to the Python Flask backend. This avoids CORS preflight
  // headaches, works the same in dev and on Vercel, and lets us swap
  // backend URLs by just editing an env var (no code change).
  //
  // ROOT routing (/) is handled by the redirect in app/page.tsx — we
  // intentionally avoid rewriting `/` itself because Next.js dev mode
  // injects React hydration scripts into rewrite responses, which the
  // legacy vanilla-JS code chokes on.
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${BACKEND_URL}/api/:path*` },
    ];
  },
};

export default nextConfig;
