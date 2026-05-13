// Diagnostic endpoint — confirm Vercel actually picked up the
// BACKEND_URL environment variable. Visit /api/debug from the browser
// to see what the server sees.
//
// Next.js route handlers take priority over next.config.mjs rewrites,
// so /api/debug never gets proxied to the Python backend — it always
// hits this file.

export const dynamic = 'force-dynamic';

export async function GET() {
  const backendUrl = process.env.BACKEND_URL;
  return Response.json({
    backendUrl: backendUrl || '(NOT SET — env var missing)',
    hasBackendUrl: typeof backendUrl === 'string' && backendUrl.length > 0,
    rewriteTarget: `${backendUrl || 'http://localhost:5000'}/api/<anything>`,
    nodeEnv: process.env.NODE_ENV,
    vercelEnv: process.env.VERCEL_ENV || '(not on Vercel)',
    vercelUrl: process.env.VERCEL_URL || '(not on Vercel)',
    deploymentTimestamp: new Date().toISOString(),
    note: 'If hasBackendUrl=false, set BACKEND_URL in Vercel project Environment Variables and redeploy WITHOUT build cache.',
  });
}
