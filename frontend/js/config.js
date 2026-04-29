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

    // Sentinel that a build step can replace. If not replaced, it stays as the
    // literal string and we ignore it.
    var fromBuild = '__API_BASE__';
    if (fromBuild === ('__API_' + 'BASE__')) fromBuild = '';

    var apiBase = existing.API_BASE || fromBuild || '';

    // Derive WebSocket base from API_BASE for future use (not used today on FE)
    var wsBase = '';
    if (apiBase) {
        wsBase = apiBase.replace(/^http/i, 'ws');
    }

    window.APP_CONFIG = Object.assign({}, existing, {
        API_BASE: apiBase.replace(/\/+$/, ''),  // strip trailing slash
        WS_BASE:  wsBase.replace(/\/+$/, ''),
    });
})();
