/**
 * APP_CONFIG — runtime config for the frontend.
 *
 * Loaded BEFORE app.js so every fetch() can prefix the backend URL from here.
 *
 * Priority order (first non-empty wins):
 *   1. window.APP_CONFIG.API_BASE that someone sets before this file loads
 *   2. Vercel/Netlify build-time replacement of the __API_BASE__ sentinel below
 *   3. "" (empty string) → same-origin, useful for local dev when FE+BE are on
 *      the same host (`cd frontend && npm start` on port 3000)
 *
 * === Deploy on Vercel ===
 * Set the backend URL in one of two ways:
 *   A) Edit the line below before pushing:
 *        API_BASE: 'https://your-backend.example.com'
 *   B) Use a build step that substitutes `__API_BASE__` with Vercel env
 *      var at deploy time (see vercel.json notes in repo root).
 *
 * The API_BASE should have NO trailing slash.
 */
(function () {
    var existing = (typeof window !== 'undefined' && window.APP_CONFIG) || {};

    // ╔════════════════════════════════════════════════════════════════════╗
    // ║  EDIT THIS LINE BEFORE DEPLOYING TO VERCEL                         ║
    // ║                                                                    ║
    // ║  Set to your ngrok / cloudflared tunnel URL pointing at your       ║
    // ║  local Express server (port 3000), e.g.:                           ║
    // ║      var BACKEND_URL = 'https://abcd-123-45.ngrok-free.app';       ║
    // ║                                                                    ║
    // ║  Leave as '' for LOCAL DEV (FE served by Express on same origin).  ║
    // ╚════════════════════════════════════════════════════════════════════╝
    var BACKEND_URL = '';

    // Sentinel that a build step can replace. If not replaced, it stays as the
    // literal string and we ignore it.
    var fromBuild = '__API_BASE__';
    if (fromBuild === ('__API_' + 'BASE__')) fromBuild = '';

    var apiBase = existing.API_BASE || BACKEND_URL || fromBuild || '';

    // Derive WebSocket base from API_BASE for future use (not used today on FE)
    var wsBase = '';
    if (apiBase) {
        wsBase = apiBase.replace(/^http/i, 'ws');
    }

    window.APP_CONFIG = Object.assign({}, existing, {
        API_BASE: apiBase.replace(/\/+$/, ''),  // strip trailing slash
        WS_BASE:  wsBase.replace(/\/+$/, ''),
    });

    // ─── ngrok bypass ────────────────────────────────────────────────────
    // ngrok free-tier shows an HTML "Visit Site" interstitial on the FIRST
    // request from each browser. fetch() then gets HTML instead of JSON →
    // parse error. We monkey-patch window.fetch to always inject the
    // `ngrok-skip-browser-warning` header (any non-empty value works) on
    // requests going to our backend, which makes ngrok serve the real
    // response immediately. Only active when API_BASE looks like an ngrok
    // URL — no effect on local dev.
    if (window.APP_CONFIG.API_BASE && /ngrok|trycloudflare|cloudflare/i.test(window.APP_CONFIG.API_BASE)) {
        var _origFetch = window.fetch.bind(window);
        window.fetch = function (input, init) {
            init = init || {};
            init.headers = new Headers(init.headers || {});
            init.headers.set('ngrok-skip-browser-warning', 'true');
            return _origFetch(input, init);
        };
        console.log('[Config] ngrok-bypass header active for', window.APP_CONFIG.API_BASE);
    }
})();
