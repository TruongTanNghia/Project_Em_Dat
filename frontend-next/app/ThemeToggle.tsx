'use client';

import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark';

/* Theme toggle — small icon button in the registry header. The visible
   icon mirrors the theme the user would switch TO (sun while dark, moon
   while light), matching how Stripe / Linear / GitHub do it. State syncs
   with the html[data-theme] attribute that the FOUC bootstrap in
   layout.tsx already set on first paint. */
export function ThemeToggle() {
  /* SSR-safe initial: pick whatever the FOUC script wrote, default light. */
  const [theme, setTheme] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const current =
      (document.documentElement.getAttribute('data-theme') as Theme | null) ||
      'light';
    setTheme(current);
    setMounted(true);
  }, []);

  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem('ada-theme', next);
    } catch {}
    setTheme(next);
  };

  /* Render a stable placeholder until mounted so the server-rendered
     markup matches the client's first render (avoids hydration mismatch
     warnings even with suppressHydrationWarning on). */
  const label = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label={label}
      title={label}
      data-theme-state={mounted ? theme : 'idle'}
    >
      <span className="theme-toggle-track" aria-hidden>
        <span className="theme-toggle-knob">
          {/* Sun glyph — shown when current theme is light (rest position) */}
          <svg
            className="theme-icon theme-icon-sun"
            viewBox="0 0 24 24"
            width="13"
            height="13"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
          </svg>
          {/* Moon glyph — shown when current theme is dark */}
          <svg
            className="theme-icon theme-icon-moon"
            viewBox="0 0 24 24"
            width="13"
            height="13"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        </span>
      </span>
      <span className="theme-toggle-label">
        {mounted ? (theme === 'dark' ? 'Dark' : 'Light') : '—'}
      </span>
    </button>
  );
}
