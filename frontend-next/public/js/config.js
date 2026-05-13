/**
 * APP_CONFIG — runtime config for the frontend.
 *
 * With the Next.js wrapper, /api/* requests are PROXIED by Next.js (via
 * `rewrites()` in next.config.mjs) to the URL set in the BACKEND_URL env
 * variable. So from the legacy app's point of view, all fetches should
 * go to the SAME ORIGIN (no API_BASE prefix needed).
 *
 *   Local dev:  set BACKEND_URL=http://localhost:5000 in .env.local
 *   Vercel:     set BACKEND_URL=https://xxxx.ngrok-free.app in project env
 *
 * API_BASE is kept at '' here so legacy fetch() calls produce paths like
 * /api/predict-brain that hit the Next.js server, which then rewrites
 * to the configured backend.
 */
(function () {
    var existing = (typeof window !== 'undefined' && window.APP_CONFIG) || {};
    var apiBase  = existing.API_BASE || '';

    window.APP_CONFIG = Object.assign({}, existing, {
        API_BASE: apiBase.replace(/\/+$/, ''),
        WS_BASE:  apiBase ? apiBase.replace(/^http/i, 'ws').replace(/\/+$/, '') : '',
    });

    // ─── ngrok bypass header on EVERY fetch ──────────────────────────────
    // ngrok free-tier serves an HTML interstitial on the first request from
    // each browser ("Visit Site" page), which makes fetch() get HTML
    // instead of JSON. Sending `ngrok-skip-browser-warning: true` (any
    // non-empty value) skips the page entirely.
    //
    // We add the header unconditionally on every fetch — harmless when the
    // backend isn't ngrok, and Next.js rewrites forward all request
    // headers to the upstream backend, so it always arrives.
    var _origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        init = init || {};
        var h = new Headers(init.headers || {});
        if (!h.has('ngrok-skip-browser-warning')) {
            h.set('ngrok-skip-browser-warning', 'true');
        }
        init.headers = h;
        return _origFetch(input, init);
    };
})();
