/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow ngrok URL + large API payloads (NIfTI uploads ~30MB).
  experimental: {
    largePageDataBytes: 50 * 1024 * 1024,
  },

  // Rewrite the root URL to serve the legacy index.html from /public.
  // We keep the existing vanilla-JS Medical AI app intact and just wrap it
  // in a Next.js shell so Vercel can deploy it natively.
  async rewrites() {
    return [
      { source: '/', destination: '/legacy.html' },
    ];
  },

  // Disable image optimization (we serve raw GLB / PNG / base64 images
  // directly — no Next/Image usage in the legacy app).
  images: { unoptimized: true },

  // We control caching via the assets themselves (public/ folder is
  // already long-cached by Vercel automatically).
};

export default nextConfig;
