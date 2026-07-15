'use client';

import { useEffect } from 'react';

/* ScrollFX — landing page scroll enhancements:
 * (1) Reveal-on-scroll: any element with .fx-reveal fades + slides up
 *     when it enters viewport. Staggered via --fx-delay CSS var.
 * (2) Animated counters: any element with .fx-count animates from 0
 *     to its data-target when visible. Supports int + suffix ("6",
 *     "24M", "<2s", "180"). Respects prefers-reduced-motion.
 * (3) Reading progress bar: fixed top strip showing scroll %.
 * (4) Section active TOC highlight: .toc-list a matching hash.
 * (5) Parallax hero: subtle mouse-driven tilt on .scan figure
 *     (mimics 3D depth without heavy WebGL cost).
 *
 * Client-only. Boot on mount. Cleans up observers on unmount. */
export function ScrollFX() {
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ── 1 · Reveal on scroll ─────────────────────────────
    const reveals = document.querySelectorAll<HTMLElement>('.fx-reveal');
    const revealObserver = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('fx-in');
            revealObserver.unobserve(e.target);
          }
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -8% 0px' }
    );
    reveals.forEach((el, i) => {
      if (!el.style.getPropertyValue('--fx-delay')) {
        el.style.setProperty('--fx-delay', `${Math.min(i * 30, 240)}ms`);
      }
      revealObserver.observe(el);
    });

    // ── 2 · Animated counters ────────────────────────────
    const counters = document.querySelectorAll<HTMLElement>('.fx-count');
    const parseTarget = (raw: string) => {
      // "6" → {n: 6, prefix: '', suffix: ''}
      // "24M" → {n: 24, prefix: '', suffix: 'M'}
      // "<2s" → {n: 2, prefix: '<', suffix: 's'}
      // "180" → {n: 180}
      const m = raw.match(/^([<>~]?)(\d+(?:\.\d+)?)(.*)$/);
      if (!m) return { n: 0, prefix: '', suffix: raw };
      return {
        n: parseFloat(m[2]),
        prefix: m[1] || '',
        suffix: m[3] || '',
      };
    };
    const easeOutExpo = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
    const animate = (el: HTMLElement) => {
      const target = el.dataset.target || el.textContent || '0';
      const parsed = parseTarget(target);
      if (reduce) {
        el.textContent = target;
        return;
      }
      const duration = 1200;
      const start = performance.now();
      const isInt = Number.isInteger(parsed.n);
      const step = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = easeOutExpo(t);
        const value = parsed.n * eased;
        el.textContent = parsed.prefix + (isInt ? Math.round(value).toString() : value.toFixed(1)) + parsed.suffix;
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    const countObserver = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            animate(e.target as HTMLElement);
            countObserver.unobserve(e.target);
          }
        }
      },
      { threshold: 0.4 }
    );
    counters.forEach((el) => {
      // Preserve original as data-target if not already set.
      if (!el.dataset.target) el.dataset.target = el.textContent?.trim() || '0';
      // Set to 0 initially (with prefix/suffix preserved as blank until animation)
      const parsed = parseTarget(el.dataset.target);
      el.textContent = parsed.prefix + '0' + parsed.suffix;
      countObserver.observe(el);
    });

    // ── 3 · Reading progress bar (fixed top) ─────────────
    let progressEl: HTMLDivElement | null = null;
    if (!document.getElementById('fx-progress')) {
      progressEl = document.createElement('div');
      progressEl.id = 'fx-progress';
      progressEl.setAttribute('aria-hidden', 'true');
      document.body.appendChild(progressEl);
    } else {
      progressEl = document.getElementById('fx-progress') as HTMLDivElement;
    }
    const updateProgress = () => {
      const doc = document.documentElement;
      const total = doc.scrollHeight - doc.clientHeight;
      const pct = total > 0 ? (doc.scrollTop / total) * 100 : 0;
      if (progressEl) progressEl.style.width = `${pct}%`;
    };
    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress, { passive: true });

    // ── 4 · Parallax on hero scan figure ──────────────────
    let scanEl: HTMLElement | null = null;
    let scanBounds: DOMRect | null = null;
    const handleMouse = (e: MouseEvent) => {
      if (!scanEl || !scanBounds) return;
      if (reduce) return;
      const cx = scanBounds.left + scanBounds.width / 2;
      const cy = scanBounds.top + scanBounds.height / 2;
      const dx = (e.clientX - cx) / scanBounds.width;
      const dy = (e.clientY - cy) / scanBounds.height;
      // Tilt max 6deg — enough to feel 3D, not enough to distort readability
      const rx = (dy * -6).toFixed(2);
      const ry = (dx * 6).toFixed(2);
      scanEl.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    };
    const setupParallax = () => {
      scanEl = document.querySelector<HTMLElement>('.scan');
      if (!scanEl) return;
      scanBounds = scanEl.getBoundingClientRect();
      scanEl.style.willChange = 'transform';
      scanEl.style.transition = 'transform 200ms cubic-bezier(0.16, 1, 0.3, 1)';
    };
    setupParallax();
    window.addEventListener('mousemove', handleMouse);
    window.addEventListener('resize', () => {
      scanBounds = scanEl?.getBoundingClientRect() ?? null;
    });

    // Cleanup on unmount
    return () => {
      revealObserver.disconnect();
      countObserver.disconnect();
      window.removeEventListener('scroll', updateProgress);
      window.removeEventListener('resize', updateProgress);
      window.removeEventListener('mousemove', handleMouse);
    };
  }, []);

  return null; // Behavior-only component
}
