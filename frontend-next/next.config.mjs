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

  // API proxy: every /api/* request from the legacy app gets forwarded
  // to the Python Flask backend at BACKEND_URL. Set BACKEND_URL in
  // .env.local (dev) or in Vercel project Environment Variables (prod).
  //
  // ROOT (/) is handled by app/page.tsx — the beautiful landing page
  // with 4 module cards. Each card links to /legacy.html#<module-id>.
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${BACKEND_URL}/api/:path*` },
    ];
  },
};

export default nextConfig;
