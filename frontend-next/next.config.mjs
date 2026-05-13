/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable image optimization (we serve raw GLB / PNG / base64 images
  // directly — no Next/Image usage in the legacy app).
  images: { unoptimized: true },

  // NOTE on routing: we use a server-side redirect in app/page.tsx
  // (`redirect('/legacy.html')`) rather than next.config rewrites.
  // Reason: rewrites cause Next.js dev mode to inject hydration scripts
  // (self.__next_f.push(...)) into the response stream alongside the
  // static HTML, which the legacy vanilla-JS code then fails to JSON.parse.
  // A real redirect makes the browser issue a fresh request that bypasses
  // the React rendering layer entirely.
};

export default nextConfig;
