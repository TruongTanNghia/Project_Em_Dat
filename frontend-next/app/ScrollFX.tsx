'use client';

import { useEffect } from 'react';

/* ScrollFX — landing page scroll enhancements:
 * (1) Reveal-on-scroll: any element with .fx-reveal fades + slides up
 *     when it enters viewport. Staggered via --fx-delay CSS var.
 * (2) Animated counters: any element with .fx-count animates from 0
 *     to its data-target when visible. Supports int + suffix ("6",
 *     "24M", "<2s", "180"). Respects prefers-reduced-motion.
 * (3) Reading progress bar: fixed top strip showing scroll %.
 * (4) Lung anatomy glass: flip lungs.glb materials to translucent
 *     after the GLB loads — reads as x-ray / anatomical glass.
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

    // ── 4 · Lung anatomy glass — flip lungs.glb materials to translucent
    // so the model reads like the brain (anatomical x-ray look) instead
    // of opaque flesh. If the GLB has bronchi / vessel geometry inside,
    // they'll show through; if not, it renders as ghosted glass — either
    // way better than the opaque default.
    const lungMv = document.querySelector<HTMLElement>(
      '.case-visual-3d model-viewer[src*="lungs"]'
    );
    const applyLungGlass = () => {
      const materials = (lungMv as unknown as { model?: { materials?: unknown[] } })?.model?.materials;
      if (!materials?.length) return;
      materials.forEach((mat) => {
        try {
          const m = mat as {
            setAlphaMode: (mode: string) => void;
            setDoubleSided?: (b: boolean) => void;
            pbrMetallicRoughness: {
              baseColorFactor: number[];
              setBaseColorFactor: (f: number[]) => void;
            };
          };
          m.setAlphaMode('BLEND');
          m.setDoubleSided?.(true);
          const [r, g, b] = m.pbrMetallicRoughness.baseColorFactor;
          // Warmer tint + heavy transparency for glass-anatomy read
          m.pbrMetallicRoughness.setBaseColorFactor([
            Math.min(1, r * 0.85 + 0.15),
            Math.min(1, g * 0.55 + 0.05),
            Math.min(1, b * 0.55 + 0.05),
            0.32,
          ]);
        } catch {
          /* material shape varies by GLB — skip on API mismatch */
        }
      });
    };
    if (lungMv) {
      lungMv.addEventListener('load', applyLungGlass);
      // Re-try in case load already fired before this effect mounted
      const retryTimer = window.setTimeout(applyLungGlass, 1500);
      const retryTimer2 = window.setTimeout(applyLungGlass, 4000);
      // Store timers on the element for cleanup
      (lungMv as HTMLElement & { __lungTimers?: number[] }).__lungTimers = [retryTimer, retryTimer2];
    }

    // Spine tint intentionally OFF — the new spine.glb (skull + full
    // column with baked annotations, colored markers) already has the
    // right materials. User asked to preserve the original colors.

    // Cleanup on unmount
    return () => {
      revealObserver.disconnect();
      countObserver.disconnect();
      window.removeEventListener('scroll', updateProgress);
      window.removeEventListener('resize', updateProgress);
      if (lungMv) {
        lungMv.removeEventListener('load', applyLungGlass);
        const timers = (lungMv as HTMLElement & { __lungTimers?: number[] }).__lungTimers;
        timers?.forEach((t) => window.clearTimeout(t));
      }
    };
  }, []);

  return null; // Behavior-only component
}
