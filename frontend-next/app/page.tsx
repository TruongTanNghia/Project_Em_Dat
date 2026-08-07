import Link from 'next/link';
import Image from 'next/image';
import { ThemeToggle } from './ThemeToggle';
import { ScrollFX } from './ScrollFX';
import { SpineViewer } from './SpineViewer';

/* ─── BraTS lesion overlay: irregular organic mass shapes, not
       targeting reticles. NCR core (solid red-orange blob), ET
       enhancing ring (cobalt donut wrapping the core), ED edema
       (large diffuse amber halo with infiltrative fingers).
       Each layer is a bezier-smoothed closed path through N
       points at noise-perturbed radii — deterministic per-seed
       so the same lesion always renders the same shape. ──────── */

type Lesion = {
  cx: number; cy: number;
  rEd: number;       // edema halo radius
  rEt: number;       // enhancing ring outer radius
  rNcr: number;      // necrotic core radius
  rot?: number;
  seed?: number;
};

// Deterministic 1-D noise so SSR/client agree on shapes
function noise1d(seed: number, i: number): number {
  const x = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

// Catmull-Rom closed bezier through points at radius r * (1 ± amp * noise)
function blobPath(r: number, pts: number, seed: number, amp: number, ar = 1): string {
  const TAU = Math.PI * 2;
  const points = Array.from({ length: pts }, (_, i) => {
    const a = (i / pts) * TAU + noise1d(seed, i + 91) * 0.25;
    const rad = r * (1 - amp + 2 * amp * noise1d(seed, i));
    return { x: Math.cos(a) * rad, y: Math.sin(a) * rad * ar };
  });
  let d = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
  for (let i = 0; i < pts; i++) {
    const p0 = points[(i - 1 + pts) % pts];
    const p1 = points[i];
    const p2 = points[(i + 1) % pts];
    const p3 = points[(i + 2) % pts];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)} ${c2x.toFixed(2)} ${c2y.toFixed(2)} ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }
  return d + ' Z';
}

/* ─── NoduleMarker — small lung nodule (single-class segmentation):
       solid core + soft glow halo. Simpler than BraTS tumor because
       LIDC-IDRI uses just 1 class (nodule yes/no). */
function NoduleMarker({ size = 48, seed = 3 }: { size?: number; seed?: number }) {
  const half = size / 2;
  const haloPath = blobPath(half * 0.92, 12, seed + 1, 0.22, 0.95);
  const corePath = blobPath(half * 0.42, 10, seed + 4, 0.28);
  const blurId = `nod-blur-${seed}`;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`-${half} -${half} ${size} ${size}`}
      className="lesion-marker"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <defs>
        <filter id={blurId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={size * 0.02} />
        </filter>
        <radialGradient id={`nod-grad-${seed}`} cx="40%" cy="40%" r="65%">
          <stop offset="0%"  stopColor="oklch(0.85 0.18 130)" />
          <stop offset="100%" stopColor="oklch(0.55 0.16 125)" />
        </radialGradient>
      </defs>
      <path d={haloPath}
            fill="oklch(0.70 0.14 130 / 0.55)"
            filter={`url(#${blurId})`} />
      <path d={corePath}
            fill={`url(#nod-grad-${seed})`} />
    </svg>
  );
}

/* ─── LesionMarker — 3-layer BraTS tumor blob (ED halo + ET ring +
       NCR core) shaped like a real segmentation, not a circle.
       Goes inside a <model-viewer> hotspot so it follows rotation. */
function LesionMarker({
  size = 88,
  seed = 7,
  edScale = 0.96,
  etScale = 0.55,
  ncrScale = 0.30,
}: {
  size?: number;
  seed?: number;
  edScale?: number;
  etScale?: number;
  ncrScale?: number;
}) {
  const half = size / 2;
  const edPath  = blobPath(half * edScale,  14, seed + 1, 0.32, 0.88);
  const etOut   = blobPath(half * etScale,  12, seed + 2, 0.22);
  const etIn    = blobPath(half * etScale * 0.58, 10, seed + 3, 0.30);
  const ncrPath = blobPath(half * ncrScale, 10, seed + 4, 0.28);
  const blurId = `lesion-blur-${seed}`;
  const coreBlurId = `lesion-core-blur-${seed}`;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`-${half} -${half} ${size} ${size}`}
      className="lesion-marker"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <defs>
        <filter id={blurId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={size * 0.025} />
        </filter>
        <filter id={coreBlurId} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation={size * 0.008} />
        </filter>
        <radialGradient id={`ed-grad-${seed}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%"  stopColor="oklch(0.78 0.16 70 / 0.75)" />
          <stop offset="60%" stopColor="oklch(0.78 0.16 70 / 0.45)" />
          <stop offset="100%" stopColor="oklch(0.78 0.16 70 / 0)" />
        </radialGradient>
        <radialGradient id={`ncr-grad-${seed}`} cx="40%" cy="40%" r="65%">
          <stop offset="0%"  stopColor="oklch(0.72 0.22 25)" />
          <stop offset="100%" stopColor="oklch(0.50 0.22 25)" />
        </radialGradient>
      </defs>

      {/* ED — large diffuse amber halo (edema) */}
      <path d={edPath}
            fill={`url(#ed-grad-${seed})`}
            filter={`url(#${blurId})`} />

      {/* ET — enhancing tumor ring: outer cobalt minus inner cutout */}
      <path d={`${etOut} ${etIn}`}
            fill="oklch(0.55 0.22 268 / 0.85)"
            fillRule="evenodd" />

      {/* NCR — necrotic core, red, slight blur for soft edges */}
      <path d={ncrPath}
            fill={`url(#ncr-grad-${seed})`}
            filter={`url(#${coreBlurId})`} />
    </svg>
  );
}

function LesionOverlay({ lesions, style }: { lesions: Lesion[]; style?: React.CSSProperties }) {
  return (
    <svg
      className="lesion-overlay"
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="none"
      aria-hidden="true"
      style={style}
    >
      <defs>
        <filter id="edema-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="1.8" />
        </filter>
        <filter id="core-blur" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="0.45" />
        </filter>
      </defs>
      {lesions.map((l, i) => {
        const seed = l.seed ?? i * 17 + 3;
        const edPath = blobPath(l.rEd, 14, seed + 1, 0.32, 0.85);
        const etOut = blobPath(l.rEt, 12, seed + 2, 0.22);
        const etIn  = blobPath(l.rEt * 0.55, 10, seed + 3, 0.30);
        const ncrPath = blobPath(l.rNcr, 10, seed + 4, 0.28);
        return (
          <g key={i} transform={`translate(${l.cx} ${l.cy}) rotate(${l.rot ?? 0})`}>
            {/* ED edema — large diffuse amber blob with infiltrative bumps */}
            <path
              d={edPath}
              fill="oklch(0.72 0.16 60)"
              opacity="0.22"
              filter="url(#edema-blur)"
            />
            {/* Second edema layer for depth — slightly offset, fainter, more diffuse */}
            <path
              d={blobPath(l.rEd * 0.78, 12, seed + 5, 0.35, 0.9)}
              fill="oklch(0.78 0.14 55)"
              opacity="0.18"
              filter="url(#edema-blur)"
              transform={`translate(${noise1d(seed, 99) * 2 - 1} ${noise1d(seed, 100) * 2 - 1})`}
            />
            {/* ET enhancing ring — cobalt donut, evenodd fill */}
            <path
              d={`${etOut} ${etIn}`}
              fill="oklch(0.62 0.20 252)"
              fillRule="evenodd"
              opacity="0.78"
              filter="url(#core-blur)"
            />
            {/* NCR necrotic core — solid irregular red-orange mass */}
            <path
              className="lesion-core"
              d={ncrPath}
              fill="oklch(0.55 0.22 28)"
              opacity="0.85"
              filter="url(#core-blur)"
            />
            {/* Subtle inner highlight on NCR — gives the mass a volumetric feel */}
            <path
              d={blobPath(l.rNcr * 0.45, 8, seed + 6, 0.35)}
              fill="oklch(0.70 0.18 35)"
              opacity="0.4"
              filter="url(#core-blur)"
              transform={`translate(${-l.rNcr * 0.25} ${-l.rNcr * 0.2})`}
            />
          </g>
        );
      })}
    </svg>
  );
}

/* ─── Lung nodule overlay: small cobalt focal opacity with corner
       calipers + measurement readout. Same HUD layer convention as
       the brain lesion. ─────────────────────────────────────────── */

function NoduleOverlay({ cx, cy, r, label, seed = 11 }: { cx: number; cy: number; r: number; label: string; seed?: number }) {
  const c = r * 2.0; // caliper bracket distance from centre
  const k = r * 0.6; // bracket arm length
  // Spiculation: 6 short radial spikes at noisy angles, varying length
  const spikes = Array.from({ length: 7 }, (_, i) => {
    const a = (i / 7) * Math.PI * 2 + noise1d(seed + 70, i) * 0.4;
    const r1 = r * 0.95;
    const r2 = r * (1.4 + noise1d(seed + 80, i) * 0.6);
    return {
      x1: Math.cos(a) * r1, y1: Math.sin(a) * r1,
      x2: Math.cos(a) * r2, y2: Math.sin(a) * r2,
    };
  });
  return (
    <svg
      className="lesion-overlay"
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        <filter id="nodule-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="0.6" />
        </filter>
        <filter id="nodule-halo" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="1.6" />
        </filter>
      </defs>
      <g transform={`translate(${cx} ${cy})`}>
        {/* Ground-glass halo around the nodule (CT GGO opacity feel) */}
        <path
          d={blobPath(r * 1.7, 14, seed + 11, 0.25)}
          fill="oklch(0.65 0.18 252)"
          opacity="0.18"
          filter="url(#nodule-halo)"
        />
        {/* Solid nodule mass — irregular blob */}
        <path
          className="lesion-core"
          d={blobPath(r, 12, seed, 0.28)}
          fill="oklch(0.60 0.20 252)"
          opacity="0.80"
          filter="url(#nodule-blur)"
        />
        {/* Spiculations — short radial spikes, malignancy sign */}
        <g stroke="oklch(0.65 0.18 252)" strokeWidth="0.25" strokeLinecap="round" opacity="0.85">
          {spikes.map((s, i) => (
            <line key={i} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} />
          ))}
        </g>
        {/* Caliper brackets — clinical measurement standard */}
        <g stroke="oklch(0.95 0.04 252)" strokeWidth="0.22" fill="none">
          <path d={`M ${-c} ${-c + k} L ${-c} ${-c} L ${-c + k} ${-c}`} />
          <path d={`M ${c - k} ${-c} L ${c} ${-c} L ${c} ${-c + k}`} />
          <path d={`M ${-c} ${c - k} L ${-c} ${c} L ${-c + k} ${c}`} />
          <path d={`M ${c - k} ${c} L ${c} ${c} L ${c} ${c - k}`} />
        </g>
        {/* Measurement label */}
        <text
          x={c + 1}
          y={-c - 0.4}
          fontFamily="ui-monospace, monospace"
          fontSize="2.4"
          fontWeight="600"
          letterSpacing="0.15"
          fill="oklch(0.95 0.04 252)"
        >
          {label}
        </text>
      </g>
    </svg>
  );
}

/* ─── Hero: real anatomical brain GLB rotating, with radiologist's
       annotation pills overlayed. The model-viewer custom element is
       defined by the script loaded in app/layout.tsx. ──────────── */

function HeroScan() {
  return (
    <figure className="scan">
      <model-viewer
        src="/models/brain/human-brain.glb"
        alt="Human brain — anatomical 3D model from BraTS-2021 case 00621"
        auto-rotate
        rotation-per-second="18deg"
        camera-controls
        disable-zoom
        disable-pan
        interaction-prompt="none"
        exposure="0.9"
        shadow-intensity="0.6"
        shadow-softness="1"
        tone-mapping="neutral"
        camera-orbit="-25deg 78deg 130%"
        min-camera-orbit="auto 60deg 130%"
        max-camera-orbit="auto 100deg 130%"
        loading="eager"
        reveal="auto"
        touch-action="pan-y"
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: 'transparent',
          '--poster-color': 'transparent',
        } as React.CSSProperties}
      >
        {/* 3D hotspots — these are children of <model-viewer> and get
            positioned via data-position (x y z in the model's own
            space). They follow the brain's rotation automatically and
            hide when occluded behind geometry. Replaces the old
            fixed-position 2D overlay that didn't track rotation. */}
        {/* Primary mass — 3 stat labels stacked on the right */}
        <button
          slot="hotspot-ncr"
          className="annot hotspot"
          data-position="0.055 0.025 0.045"
          data-normal="0.6 0.3 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          NCR · <b>12.3 cm³</b>
        </button>
        <button
          slot="hotspot-et"
          className="annot hotspot"
          data-position="0.060 0.005 0.045"
          data-normal="0.7 0 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          ET ring · <b>4.2 cm³</b>
        </button>
        <button
          slot="hotspot-ed"
          className="annot hotspot"
          data-position="0.055 -0.015 0.045"
          data-normal="0.6 -0.3 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          ED edema · <b>11.9 cm³</b>
        </button>
        {/* Satellite lesion — 1 label */}
        <button
          slot="hotspot-sat"
          className="annot hotspot"
          data-position="-0.060 -0.025 0.045"
          data-normal="-0.6 -0.2 0.85"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          Satellite · <b>2.8 cm³</b>
        </button>

        {/* Lesion masses — real BraTS-style multi-layer tumors (ED
            halo + ET ring + NCR core) rendered as SVG inside hotspots
            so they ride the brain's rotation in 3D. Two distinct
            lesions: a primary right-temporal mass and a smaller left
            satellite. */}
        <span
          slot="hotspot-lesion-1"
          className="lesion-marker-wrap"
          data-position="0.045 0.015 0.045"
          data-normal="0.5 0.2 0.85"
        >
          <LesionMarker size={110} seed={7} />
        </span>
        <span
          slot="hotspot-lesion-2"
          className="lesion-marker-wrap"
          data-position="-0.045 -0.020 0.04"
          data-normal="-0.5 -0.2 0.85"
        >
          <LesionMarker size={64} seed={19} edScale={0.92} etScale={0.50} ncrScale={0.24} />
        </span>
      </model-viewer>

      {/* HUD corner stamps — these are fixed-position on the figure
          frame (deliberately don't rotate, they're the radiologist's
          monitor chrome around the actual model). */}
      <span className="scan-corner tl"><b>BraTS-2021</b> · case 00621</span>
      <span className="scan-corner tr">FLAIR · ax<br /><b>z = 76</b></span>
      <span className="scan-corner bl">3D U-Net · TF<br /><b>Dice WT 0.83</b></span>
      <span className="scan-corner br"><b>R</b> · L</span>
    </figure>
  );
}

/* ─── Per-case visuals (4 differentiated) ───────────────────────── */

function CaseVisualEEG() {
  return (
    <div className="case-visual">
      <svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <defs>
          <pattern id="grid01" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="oklch(0.22 0.008 25)" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="800" height="400" fill="url(#grid01)" />
        {/* 6 channel EEG waveforms, with a 'seizure burst' in the middle */}
        {[0, 1, 2, 3, 4, 5].map((i) => {
          const yBase = 50 + i * 56;
          // Construct a wavy path with a burst around x ~ 380-520
          let path = `M 0 ${yBase}`;
          for (let x = 0; x <= 800; x += 8) {
            const t = x / 800;
            const calm = Math.sin(t * 14 + i) * 3 + Math.sin(t * 7 + i * 2) * 2;
            const burstAmt = Math.max(0, 1 - Math.abs(t - 0.55) * 6);
            const burst = Math.sin(t * 80 + i) * 22 * burstAmt;
            const y = yBase + calm + burst;
            path += ` L ${x} ${y.toFixed(1)}`;
          }
          return (
            <path
              key={i}
              d={path}
              fill="none"
              stroke={i === 2 ? 'oklch(0.78 0.16 70)' : 'oklch(0.55 0.04 28)'}
              strokeWidth={i === 2 ? '1.4' : '0.9'}
              opacity={i === 2 ? 1 : 0.7}
            />
          );
        })}
        {/* Seizure burst highlight rectangle */}
        <rect
          className="draw"
          x="380" y="20" width="140" height="360"
          fill="none"
          stroke="oklch(0.78 0.16 70)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          style={{ ['--dash' as string]: 60, ['--delay' as string]: '0.5s' }}
        />
      </svg>
      <span className="scan-corner tl"><b>CHB-MIT</b> · chb01_03</span>
      <span className="scan-corner tr">23 ch · 256 Hz<br /><b>seizure t=92s</b></span>
      <span className="scan-corner bl">CNN+BiGRU<br /><b>p = 0.93</b></span>
      <span className="scan-corner br">0.5–40 Hz</span>
    </div>
  );
}

function CaseVisualBrain() {
  return (
    <div className="case-visual case-visual-3d">
      <model-viewer
        src="/models/brain/human-brain.glb"
        alt="Brain anatomy with segmented tumor regions"
        auto-rotate
        rotation-per-second="12deg"
        camera-controls
        disable-zoom
        disable-pan
        interaction-prompt="none"
        exposure="0.85"
        shadow-intensity="0.5"
        tone-mapping="neutral"
        camera-orbit="35deg 75deg 130%"
        min-camera-orbit="auto 60deg 130%"
        max-camera-orbit="auto 100deg 130%"
        loading="lazy"
        reveal="auto"
        touch-action="pan-y"
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' } as React.CSSProperties}
      >
        {/* 3D-anchored BraTS lesion mass — follows brain rotation */}
        <span slot="hotspot-c2-mass" className="lesion-marker-wrap"
              data-position="0.050 0.010 0.045" data-normal="0.6 0.2 0.8">
          <LesionMarker size={86} seed={31} />
        </span>
        <span slot="hotspot-c2-sat" className="lesion-marker-wrap"
              data-position="-0.040 -0.020 0.04" data-normal="-0.5 -0.2 0.85">
          <LesionMarker size={48} seed={47} edScale={0.92} etScale={0.50} ncrScale={0.24} />
        </span>

        {/* Per-region annotation pills for the primary mass — three
            labels stacked on the right side, same convention as the
            hero brain. Volumes sum to 28.4 cm³ shown in scan-corner br. */}
        <button
          slot="hotspot-c2-ncr"
          className="annot hotspot"
          data-position="0.058 0.028 0.045"
          data-normal="0.6 0.3 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          NCR · <b>7.8 cm³</b>
        </button>
        <button
          slot="hotspot-c2-et"
          className="annot hotspot"
          data-position="0.062 0.008 0.045"
          data-normal="0.7 0 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          ET ring · <b>3.4 cm³</b>
        </button>
        <button
          slot="hotspot-c2-ed"
          className="annot hotspot"
          data-position="0.058 -0.012 0.045"
          data-normal="0.6 -0.3 0.8"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          ED edema · <b>15.2 cm³</b>
        </button>
        {/* Satellite lesion label */}
        <button
          slot="hotspot-c2-sat-label"
          className="annot hotspot"
          data-position="-0.055 -0.028 0.045"
          data-normal="-0.6 -0.2 0.85"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          Satellite · <b>2.0 cm³</b>
        </button>
      </model-viewer>
      {/* Tiny class-legend chip (HUD-style key, mono spec) */}
      <span className="lesion-legend">
        <span><i className="dot-ncr" /> NCR</span>
        <span><i className="dot-et" /> ET</span>
        <span><i className="dot-ed" /> ED</span>
      </span>
      <span className="scan-corner tl"><b>BraTS-2020</b> · 4-channel</span>
      <span className="scan-corner tr">FLAIR/T1/T1c/T2<br /><b>128³ TTA</b></span>
      <span className="scan-corner bl">NCR · ED · ET<br /><b>Marching cubes</b></span>
      <span className="scan-corner br">vol <b>28.4 cm³</b></span>
    </div>
  );
}

function CaseVisualLung() {
  return (
    <div className="case-visual case-visual-3d">
      <model-viewer
        src="/models/lungs.glb"
        alt="Lung anatomy 3D model with highlighted nodule"
        auto-rotate
        rotation-per-second="10deg"
        camera-controls
        disable-zoom
        disable-pan
        interaction-prompt="none"
        exposure="1.0"
        shadow-intensity="0.4"
        tone-mapping="neutral"
        /* Frontal view so BOTH lung halves are visible; slight tilt
           down so the upper lobes read clearly. 105% distance so lungs
           fill the viewport (was 150% — model ended up too small/off-
           frame relative to Brain which uses 130%). */
        camera-orbit="0deg 78deg 105%"
        min-camera-orbit="-30deg 65deg 105%"
        max-camera-orbit="30deg 95deg 105%"
        /* eager so 25MB starts fetching on page load — lazy was
           firing IO but model never appeared on screen */
        loading="eager"
        reveal="auto"
        touch-action="pan-y"
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' } as React.CSSProperties}
      >
        {/* 3D-anchored nodule marker — small mass in the right upper
            lobe. Follows lung rotation. */}
        <span
          slot="hotspot-lung-nodule"
          className="nodule-marker-wrap"
          data-position="0.040 0.035 0.020"
          data-normal="0.7 0.4 0.6"
        >
          <NoduleMarker size={56} />
        </span>
        <button
          slot="hotspot-lung-label"
          className="annot hotspot"
          data-position="0.055 0.050 0.020"
          data-normal="0.7 0.4 0.6"
          data-visibility-attribute="visible"
        >
          <span className="annot-tick" />
          ø <b>18 mm</b> · upper lobe
        </button>
      </model-viewer>
      {/* Malignancy meter — vertical mini-gauge bottom-right corner */}
      <div className="mal-meter">
        <span className="mal-meter-label">MALIGNANCY</span>
        <div className="mal-meter-bars">
          {[1, 2, 3, 4, 5].map((n) => (
            <div key={n} className={`mal-bar${n <= 4 ? ' on' : ''}`} />
          ))}
        </div>
        <span className="mal-meter-score"><b>4</b>/5</span>
      </div>
      <span className="scan-corner tl"><b>LIDC-IDRI</b> · ax slice 84</span>
      <span className="scan-corner tr">DeepLabV3<br /><b>ø 18 mm</b></span>
      <span className="scan-corner bl">solid · spiculated</span>
      <span className="scan-corner br"><b>R</b> upper lobe</span>
    </div>
  );
}

function CaseVisualBlood() {
  return (
    <div className="case-visual" style={{ background: 'oklch(0.13 0.006 25)' }}>
      <svg viewBox="0 0 800 550" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <rect width="800" height="550" fill="oklch(0.13 0.006 25)" />
        {/* Lab report header */}
        <text x="50" y="60" fontFamily="ui-monospace, monospace" fontSize="13"
              fill="oklch(0.60 0.005 25)" letterSpacing="3">
          PATIENT · BC-280514 · COMPLETE BLOOD COUNT
        </text>
        <line x1="50" y1="78" x2="750" y2="78" stroke="oklch(0.22 0.008 25)" strokeWidth="1" />

        {/* Table rows */}
        {[
          { name: 'Hemoglobin',      val: '11.2 g/dL',  ref: '13.5–17.5',  bad: true,  y: 120 },
          { name: 'RBC',             val: '4.1 ×10⁶/µL', ref: '4.5–5.9',   bad: true,  y: 158 },
          { name: 'WBC',             val: '13.8 ×10³',   ref: '4.0–10.0',  bad: true,  y: 196 },
          { name: 'Platelets',       val: '256 ×10³',    ref: '150–400',   bad: false, y: 234 },
          { name: 'Glucose (fast.)', val: '142 mg/dL',   ref: '70–99',     bad: true,  y: 272 },
          { name: 'HDL',             val: '38 mg/dL',    ref: '>40',       bad: true,  y: 310 },
          { name: 'LDL',             val: '168 mg/dL',   ref: '<100',      bad: true,  y: 348 },
          { name: 'Triglycerides',   val: '189 mg/dL',   ref: '<150',      bad: true,  y: 386 },
          { name: 'HbA1c',           val: '6.8 %',       ref: '<5.7',      bad: true,  y: 424 },
        ].map((row) => (
          <g key={row.name}>
            <text x="50" y={row.y} fontFamily="ui-monospace, monospace" fontSize="13"
                  fill="oklch(0.78 0.008 25)" letterSpacing="0.5">
              {row.name}
            </text>
            <text x="330" y={row.y} fontFamily="ui-monospace, monospace" fontSize="14"
                  fontWeight="600"
                  fill={row.bad ? 'oklch(0.66 0.18 5)' : 'oklch(0.95 0.008 25)'}
                  letterSpacing="0.5">
              {row.val}
            </text>
            <text x="540" y={row.y} fontFamily="ui-monospace, monospace" fontSize="12"
                  fill="oklch(0.50 0.005 25)" letterSpacing="0.5">
              ref {row.ref}
            </text>
            {row.bad && (
              <g>
                <rect className="draw"
                      x="310" y={row.y - 17} width="180" height="24"
                      fill="none" stroke="oklch(0.66 0.18 5)" strokeWidth="1.2"
                      strokeDasharray="3 3"
                      style={{ ['--dash' as string]: 80, ['--delay' as string]: `${0.4 + row.y * 0.001}s` }} />
                <text x="708" y={row.y} fontFamily="ui-monospace, monospace" fontSize="11"
                      fontWeight="600" fill="oklch(0.66 0.18 5)" letterSpacing="1">
                  ▲
                </text>
              </g>
            )}
          </g>
        ))}
      </svg>
      <span className="scan-corner br">CBC · LIPID · GLUCOSE<br /><b>7 / 9 flagged</b></span>
    </div>
  );
}

/* ─── CaseVisualBreast: Sketchfab iframe embed of "Mammary Gland Cross
       Section" by arloopa. Uses the model's baked-in annotations so
       viewers can click 1-7 pins to focus on lobes, ducts, fatty tissue
       (matches the interactive spine card's UX). Corner labels are
       overlaid on top of the iframe. Credit link at the bottom-right
       satisfies Sketchfab attribution terms.
       Kept the old SVG version below as CaseVisualBreastSvg for
       reference. Currently unused. ─────────────────────────────────── */

function CaseVisualBreast() {
  const SF_EMBED =
    'https://sketchfab.com/models/725520fe8a0b4c74bfed69b0c4d31da3/embed'
    + '?autospin=0.15'      // gentle rotation — ~8°/sec
    + '&autostart=1'
    + '&preload=1'
    + '&transparent=1'      // no white background — show our dark card
    + '&ui_theme=dark'
    + '&ui_infos=0'         // hide top title + author line
    + '&ui_stop=0'          // hide play button
    + '&ui_watermark=0'     // hide watermark link (free tier ignores)
    + '&ui_help=0'          // hide help "?" button
    + '&ui_settings=0'      // hide gear icon
    + '&ui_inspector=0'
    + '&ui_ar=0'            // hide AR button
    + '&ui_vr=0'
    + '&ui_hint=0'          // hide "Click to interact" hint
    + '&ui_fullscreen=0'
    + '&ui_annotations=1'   // KEEP the numbered pins clickable
    + '&camera=0';          // skip intro camera animation
  return (
    <div className="case-visual case-visual-embed">
      <iframe
        title="Mammary Gland Cross Section — 3D anatomy"
        src={SF_EMBED}
        allow="autoplay; fullscreen; xr-spatial-tracking"
        allowFullScreen
        loading="lazy"
        className="case-embed-frame"
      />
      <span className="scan-corner tl"><b>BUSI</b> · BreastCancerSegmentor</span>
      <span className="scan-corner tr">U-Net 2.16M<br /><b>+ BiomedCLIP</b></span>
      <span className="scan-corner bl">DenseCRF refine<br /><b>BI-RADS</b></span>
      <span className="scan-corner br">
        <a href="https://sketchfab.com/3d-models/mammary-gland-cross-section-725520fe8a0b4c74bfed69b0c4d31da3"
           target="_blank" rel="noopener noreferrer"
           className="case-embed-credit">
          Model · <b>arloopa</b>
        </a>
      </span>
    </div>
  );
}

/* Legacy SVG mockup — kept for reference, currently unused since we
   embed the Sketchfab model above. Delete if the embed proves stable. */
function CaseVisualBreastSvg() {
  const maskPath = blobPath(54, 18, 31, 0.15, 0.78);
  /* A handful of hyperechoic bright specks scattered around the mass —
     bright dots reading as reflected tissue interfaces. Deterministic
     from a seed so the composition doesn't change on hot-reload. */
  const specks = Array.from({ length: 42 }, (_, i) => {
    const a = (i * 137.508) % 360;
    const r = 15 + ((i * 23) % 130);
    const rad = (a * Math.PI) / 180;
    return {
      x: Math.cos(rad) * r,
      y: Math.sin(rad) * r * 0.85,
      s: 0.4 + ((i * 7) % 10) / 10,
      op: 0.25 + ((i * 11) % 60) / 100,
    };
  });

  return (
    <div className="case-visual">
      <svg viewBox="0 0 800 550" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <defs>
          {/* Dense random speckle noise — the defining texture of any
              real ultrasound frame. Turbulence + high-contrast color
              matrix gives the grainy salt-and-pepper look. */}
          <filter id="us-speckle" x="0" y="0" width="100%" height="100%">
            <feTurbulence type="fractalNoise" baseFrequency="1.4" numOctaves="2" seed="7" result="noise" />
            <feColorMatrix in="noise" values="0 0 0 0 0.85  0 0 0 0 0.82  0 0 0 0 0.78  0 0 0 1.6 -0.7" />
          </filter>
          {/* Depth-based falloff — bright near the probe (top), fades
              into dark posterior field as ultrasound attenuates. */}
          <linearGradient id="us-depth" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"  stopColor="oklch(0.48 0.015 25)" />
            <stop offset="35%" stopColor="oklch(0.32 0.012 25)" />
            <stop offset="70%" stopColor="oklch(0.16 0.008 25)" />
            <stop offset="100%" stopColor="oklch(0.09 0.005 25)" />
          </linearGradient>
          {/* Hyperechoic tissue cloud — bright around the mass to give
              the lesion visual context (real breast US shows fatty
              lobules surrounding a hypoechoic mass). */}
          <radialGradient id="us-tissue" cx="45%" cy="52%" r="60%">
            <stop offset="0%"   stopColor="oklch(0.58 0.015 25)" />
            <stop offset="55%"  stopColor="oklch(0.32 0.010 25)" />
            <stop offset="100%" stopColor="oklch(0.15 0.005 25)" />
          </radialGradient>
          {/* Hypoechoic mass — deep black core (cancer signature). */}
          <radialGradient id="us-mass" cx="50%" cy="50%" r="50%">
            <stop offset="0%"  stopColor="oklch(0.04 0.002 25)" />
            <stop offset="70%" stopColor="oklch(0.06 0.003 25)" />
            <stop offset="100%" stopColor="oklch(0.15 0.006 25)" />
          </radialGradient>
          {/* Coordinate grid for the 3D viewport backdrop. */}
          <pattern id="vol-grid" width="22" height="22" patternUnits="userSpaceOnUse">
            <path d="M 22 0 L 0 0 0 22" fill="none"
                  stroke="oklch(0.20 0.02 15)" strokeWidth="0.5" />
          </pattern>
          {/* 3D blob shading — muted salmon / dust rose. Reads as
              tissue, not candy. Key light upper-left, no gloss. */}
          <radialGradient id="blob-front" cx="34%" cy="30%" r="82%">
            <stop offset="0%"   stopColor="oklch(0.72 0.10 15)" />
            <stop offset="55%"  stopColor="oklch(0.48 0.13 10)" />
            <stop offset="100%" stopColor="oklch(0.24 0.08 8)" />
          </radialGradient>
          <radialGradient id="blob-mid" cx="42%" cy="40%" r="82%">
            <stop offset="0%"   stopColor="oklch(0.48 0.11 10)" />
            <stop offset="100%" stopColor="oklch(0.22 0.07 8)" />
          </radialGradient>
          <radialGradient id="blob-back" cx="50%" cy="50%" r="82%">
            <stop offset="0%"   stopColor="oklch(0.32 0.09 8)" />
            <stop offset="100%" stopColor="oklch(0.16 0.05 8)" />
          </radialGradient>
          <filter id="lesion-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" />
          </filter>
        </defs>

        {/* Backdrop */}
        <rect width="800" height="550" fill="oklch(0.07 0.005 25)" />

        {/* ═══════════════ LEFT · 2D ULTRASOUND ═══════════════ */}
        <g>
          {/* Panel container */}
          <rect x="20" y="50" width="430" height="470" rx="4"
                fill="oklch(0.08 0.005 25)"
                stroke="oklch(0.20 0.015 25)" strokeWidth="0.5" />
          {/* Depth-based US background inside the panel */}
          <rect x="22" y="52" width="426" height="466" rx="3"
                fill="url(#us-depth)" />
          {/* Speckle noise overlay — this is what makes it read as
              actual ultrasound instead of a black rectangle. */}
          <rect x="22" y="52" width="426" height="466" rx="3"
                fill="oklch(0.7 0 0)"
                filter="url(#us-speckle)"
                opacity="0.55" />
          {/* Hyperechoic tissue wash around the lesion */}
          <ellipse cx="235" cy="290" rx="140" ry="130"
                   fill="url(#us-tissue)" opacity="0.55" />
          {/* Depth ruler on the left edge — mm markers */}
          <g fontFamily="ui-monospace, monospace" fontSize="8"
             fill="oklch(0.60 0.03 25)" letterSpacing="0.3">
            {[0, 1, 2, 3, 4, 5, 6].map((i) => (
              <g key={i} transform={`translate(30 ${90 + i * 62})`}>
                <line x1="0" y1="0" x2="6" y2="0"
                      stroke="oklch(0.52 0.02 25)" strokeWidth="0.6" />
                <text x="10" y="3">{i * 1.5}cm</text>
              </g>
            ))}
          </g>

          {/* Hypoechoic mass + posterior acoustic shadow + mask overlay */}
          <g transform="translate(235 290)">
            {/* Posterior shadow — dark column BELOW the mass, a classic
                breast-US feature indicating attenuating tissue. */}
            <rect x="-52" y="42" width="104" height="140"
                  fill="oklch(0.04 0.002 25)" opacity="0.72" />
            {/* Dark hypoechoic mass */}
            <ellipse cx="0" cy="0" rx="68" ry="48" fill="url(#us-mass)" />
            <ellipse cx="-8" cy="-6" rx="46" ry="30" fill="oklch(0.03 0.002 25)" />
            {/* Bright specks scattered in the surrounding tissue */}
            <g fill="oklch(0.90 0.02 25)">
              {specks.map((s, i) => (
                <circle key={i} cx={s.x} cy={s.y} r={s.s} opacity={s.op} />
              ))}
            </g>
            {/* Mask contour — AI segmentation output, red halo + line */}
            <path d={maskPath} fill="none"
                  stroke="oklch(0.72 0.22 8)" strokeWidth="2.4"
                  filter="url(#lesion-glow)" opacity="0.55" />
            <path d={maskPath} fill="none"
                  stroke="oklch(0.82 0.22 5)" strokeWidth="1.5" />
            {/* Caliper corners */}
            <g stroke="oklch(0.92 0.06 5)" strokeWidth="1" fill="none">
              <path d="M -78 -58 L -66 -58 L -66 -46" />
              <path d="M  78 -58 L  66 -58 L  66 -46" />
              <path d="M -78  58 L -66  58 L -66  46" />
              <path d="M  78  58 L  66  58 L  66  46" />
            </g>
            <text x="82" y="-60" fontFamily="ui-monospace, monospace"
                  fontSize="11" fontWeight="600" fill="oklch(0.92 0.06 5)"
                  letterSpacing="0.5">
              ø 28.5 mm
            </text>
          </g>

          {/* Panel label — top-left */}
          <text x="40" y="76" fontFamily="ui-monospace, monospace" fontSize="10"
                fill="oklch(0.68 0.04 25)" letterSpacing="2.5" fontWeight="500">
            2D · ULTRASOUND
          </text>
        </g>

        {/* ═══════════════ RIGHT · 3D VOLUME ═══════════════ */}
        <g>
          <rect x="470" y="50" width="310" height="470" rx="4"
                fill="oklch(0.08 0.005 15)"
                stroke="oklch(0.20 0.015 15)" strokeWidth="0.5" />
          {/* Coordinate grid backdrop */}
          <rect x="472" y="52" width="306" height="466" rx="3"
                fill="url(#vol-grid)" opacity="0.55" />
          {/* Center crosshairs — subtle */}
          <line x1="625" y1="80" x2="625" y2="500"
                stroke="oklch(0.22 0.02 15)" strokeWidth="0.5" opacity="0.4" />
          <line x1="490" y1="290" x2="760" y2="290"
                stroke="oklch(0.22 0.02 15)" strokeWidth="0.5" opacity="0.4" />

          {/* Volume rendering — 3 depth bands with wireframe overlay on
              the front band so it reads as a triangulated mesh, not a
              smooth candy blob. */}
          <g transform="translate(625 290)">
            {/* Back slice */}
            <g transform="translate(-16 -12) scale(0.72)">
              <path d={maskPath} fill="url(#blob-back)" opacity="0.72" />
              <path d={maskPath} fill="none"
                    stroke="oklch(0.55 0.09 8 / 0.5)" strokeWidth="0.8" />
            </g>
            {/* Mid slice */}
            <g transform="translate(-6 -4) scale(0.88)">
              <path d={maskPath} fill="url(#blob-mid)" opacity="0.85" />
              <path d={maskPath} fill="none"
                    stroke="oklch(0.62 0.11 8 / 0.5)" strokeWidth="0.8" />
            </g>
            {/* Front slice + wireframe triangulation hint */}
            <g transform="scale(1.05)">
              <path d={maskPath} fill="url(#blob-front)" opacity="0.94" />
              {/* Interior wireframe triangles — reads as mesh facets */}
              <g fill="none" stroke="oklch(0.88 0.05 8 / 0.32)" strokeWidth="0.5">
                <path d={maskPath} />
                <path d="M -42 -32 L 20 -22 L -10 20 Z" />
                <path d="M  20 -22 L 50  10 L  20 32 Z" />
                <path d="M -42 -32 L -10 20 L -50 12 Z" />
                <path d="M -10  20 L 20  32 L   0 48 Z" />
                <path d="M -50  12 L -10 20 L -42 46 Z" />
              </g>
              {/* Tiny vertex dots at triangle corners */}
              <g fill="oklch(0.95 0.04 8)">
                {[[-42,-32],[20,-22],[50,10],[20,32],[-10,20],[-50,12],[0,48],[-42,46]].map(([x,y], i) => (
                  <circle key={i} cx={x} cy={y} r="0.8" opacity="0.65" />
                ))}
              </g>
            </g>
          </g>

          {/* Panel label */}
          <text x="486" y="76" fontFamily="ui-monospace, monospace" fontSize="10"
                fill="oklch(0.68 0.04 25)" letterSpacing="2.5" fontWeight="500">
            3D · VOLUME
          </text>
          {/* Stats — verts / vol / diameter */}
          <g transform="translate(486 94)" fontFamily="ui-monospace, monospace" fontSize="10"
             fill="oklch(0.78 0.04 25)" letterSpacing="0.5">
            <text x="0"   y="0">verts <tspan fontWeight="600" fill="oklch(0.95 0.04 25)">2814</tspan></text>
            <text x="100" y="0">vol   <tspan fontWeight="600" fill="oklch(0.95 0.04 25)">8.4 cm³</tspan></text>
            <text x="200" y="0">ø     <tspan fontWeight="600" fill="oklch(0.95 0.04 25)">28 mm</tspan></text>
          </g>
          {/* XYZ axis gizmo */}
          <g fontFamily="ui-monospace, monospace" fontSize="9" fontWeight="700"
             letterSpacing="0.5" transform="translate(486 494)">
            <rect x="0" y="0" width="18" height="14" fill="oklch(0.16 0.06 25)"
                  stroke="oklch(0.32 0.02 25)" strokeWidth="0.5" rx="2" />
            <text x="9" y="10" textAnchor="middle" fill="oklch(0.72 0.15 25)">X</text>
            <rect x="22" y="0" width="18" height="14" fill="oklch(0.16 0.06 140)"
                  stroke="oklch(0.32 0.02 25)" strokeWidth="0.5" rx="2" />
            <text x="31" y="10" textAnchor="middle" fill="oklch(0.78 0.16 140)">Y</text>
            <rect x="44" y="0" width="18" height="14" fill="oklch(0.16 0.06 235)"
                  stroke="oklch(0.32 0.02 25)" strokeWidth="0.5" rx="2" />
            <text x="53" y="10" textAnchor="middle" fill="oklch(0.72 0.14 235)">Z</text>
          </g>
        </g>

        {/* AI verdict pill — sits INSIDE the 2D panel at the bottom so
            it doesn't stack with the HTML corner labels overlaid on the
            container. */}
        <g transform="translate(56 494)">
          <rect x="0" y="0" width="378" height="20" rx="10"
                fill="oklch(0.12 0.02 5)"
                stroke="oklch(0.48 0.10 5)" strokeWidth="0.6" />
          <circle cx="14" cy="10" r="4" fill="oklch(0.72 0.22 5)" />
          <text x="26" y="14" fontFamily="ui-monospace, monospace" fontSize="11"
                fontWeight="600" fill="oklch(0.92 0.06 5)" letterSpacing="0.5">
            BENIGN · BI-RADS 3 · 84% confidence
          </text>
          <text x="298" y="14" fontFamily="ui-monospace, monospace" fontSize="10"
                fill="oklch(0.62 0.04 25)" letterSpacing="0.5">
            ensemble
          </text>
        </g>
      </svg>
      <span className="scan-corner tl"><b>BUSI</b> · BreastCancerSegmentor</span>
      <span className="scan-corner tr">U-Net 2.16M<br /><b>+ BiomedCLIP</b></span>
      <span className="scan-corner bl">DenseCRF refine<br /><b>BI-RADS</b></span>
      <span className="scan-corner br">PACS · 3D volume</span>
    </div>
  );
}

/* ─── StatsReel: a horizontal ribbon between hero and cases. Four
       numeric facts read as one continuous typographic strip with
       vertical hairlines between them, NOT as a 4-card grid (the
       absolute-banned pattern). The bottom row carries small mono
       captions for each number so the ribbon reads like a magazine
       infographic. ─────────────────────────────────────────────────── */

function StatsReel() {
  const stats = [
    { big: '06',  unit: 'modules',     caption: 'in production' },
    { big: '04',  unit: 'datasets',    caption: 'public-domain · cited' },
    { big: '24M', unit: 'params',      caption: 'combined · trained' },
    { big: '<2s', unit: 'inference',   caption: 'per case · CPU' },
  ];
  return (
    <section className="frame stats-reel fx-reveal" aria-label="By the numbers">
      <div className="stats-eyebrow">
        <span className="stats-tag">BY THE NUMBERS</span>
        <span className="stats-tag-spec">v1.1 · 2026 — pipelines verified on hold-out</span>
      </div>
      <div className="stats-row">
        {stats.map((s, i) => (
          <div key={s.big} className="stats-cell fx-reveal" data-cell={i + 1} style={{ ['--fx-delay' as string]: `${i * 90}ms` }}>
            <div className="stats-big fx-count">{s.big}</div>
            <div className="stats-unit">{s.unit}</div>
            <div className="stats-caption">{s.caption}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ─── ArchitectureFlow: 4-stage horizontal pipeline diagram on the
       drenched dark canvas (the same surface the case visuals use).
       Each node is INPUT → PIPELINE → POSTPROCESS → VIEWER, connected
       by hand-drawn arrows that animate on reveal. Sits between the
       cases section and the credits to explain how a single PACS shell
       handles 6 different pipelines. ───────────────────────────────── */

function ArchitectureFlow() {
  const stages = [
    {
      n: '01', title: 'INPUT',
      sub: 'Drag-drop',
      points: ['PNG · JPG · DICOM · NIfTI · EDF', 'Client-side validation', 'No upload to third-party'],
    },
    {
      n: '02', title: 'AI PIPELINE',
      sub: 'Backend Flask :5000',
      points: ['Load matching model · TF / PT', 'Inference + Test-Time-Aug', 'BiomedCLIP zero-shot classify'],
    },
    {
      n: '03', title: 'POSTPROCESS',
      sub: 'Mask refinement',
      points: ['DenseCRF · morph close/open', 'Largest connected component', 'Marching squares · marching cubes'],
    },
    {
      n: '04', title: 'PACS VIEWER',
      sub: 'Next.js + Three.js',
      points: ['Pan · Zoom · Ruler · Bbox', '2D ↔ 3D tabs · OrbitControls', 'Mask / Overlay / GT swap'],
    },
  ];
  return (
    <section className="frame arch-flow fx-reveal" aria-label="System architecture">
      <div className="arch-head">
        <div className="arch-eyebrow">SYSTEM</div>
        <h2 className="arch-h2">
          Một phòng đọc, sáu pipeline.
          <span className="arch-h2-sub">
            INPUT chạy qua đúng model, qua một loạt postprocess đã verify,
            rồi đổ vào một viewer chung.
          </span>
        </h2>
      </div>
      <ol className="arch-stages">
        {stages.map((s, i) => (
          <li key={s.n} className="arch-stage" data-stage={s.n}>
            <span className="arch-stage-n">{s.n}</span>
            <h3 className="arch-stage-title">{s.title}</h3>
            <div className="arch-stage-sub">{s.sub}</div>
            <ul className="arch-stage-points">
              {s.points.map((p) => <li key={p}>{p}</li>)}
            </ul>
            {i < stages.length - 1 && (
              <svg className="arch-arrow" viewBox="0 0 60 24" aria-hidden>
                <path d="M 2 12 L 50 12 M 44 6 L 50 12 L 44 18"
                      fill="none" stroke="currentColor" strokeWidth="1.4"
                      strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ─── DatasetsCredits: journal-style attribution panel. Each dataset is
       a row: name + citation on the left, mapped to one of our cases on
       the right. Read as 'sources cited at the bottom of a paper', not
       as marketing logo soup. ─────────────────────────────────────── */

function DatasetsCredits() {
  const rows = [
    {
      name: 'BraTS 2020 / 2021',
      cite: 'Bakas et al., 2018 · arXiv:1811.02629',
      kind: 'Multimodal brain MRI · 4-channel volumes',
      cases: ['Case 02'],
    },
    {
      name: 'LIDC-IDRI',
      cite: 'Armato et al., 2011 · Medical Physics 38(2)',
      kind: 'Lung CT · 1018 patients · 4-rad consensus',
      cases: ['Case 03'],
    },
    {
      name: 'CHB-MIT Scalp EEG',
      cite: 'Shoeb, 2009 · MIT PhD thesis · PhysioNet',
      kind: '23-ch scalp EEG · 24 pediatric subjects',
      cases: ['Case 01'],
    },
    {
      name: 'BUSI · Breast Ultrasound',
      cite: 'Al-Dhabyani et al., 2020 · Data in Brief 28',
      kind: 'Ultrasound · 780 imgs · 3 classes',
      cases: ['Case 06'],
    },
    {
      name: 'TotalSegmentator',
      cite: 'Wasserthal et al., 2023 · Radiology: AI 5(5)',
      kind: 'nnU-Net · 117 anatomical structures · CT',
      cases: ['Case 05'],
    },
  ];
  return (
    <section className="frame credits fx-reveal" aria-label="Datasets and research credited">
      <div className="credits-head">
        <span className="credits-eyebrow">DATASETS</span>
        <h2 className="credits-h2">
          Trained trên data thật, citation đầy đủ.
        </h2>
        <p className="credits-lede">
          Mỗi pipeline ở trên đứng trên một dataset đã peer-review.
          Không có data tự bịa, không có synthetic chỉ-để-demo.
        </p>
      </div>
      <ul className="credits-list">
        {rows.map((r) => (
          <li className="credits-row" key={r.name}>
            <div className="credits-name">{r.name}</div>
            <div className="credits-cite">{r.cite}</div>
            <div className="credits-kind">{r.kind}</div>
            <div className="credits-map">
              {r.cases.map((c) => (
                <span key={c} className="credits-pill">{c}</span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ─── Page ─────────────────────────────────────────────────────── */

export default function Home() {
  return (
    <div className="page">
      <ScrollFX />
      <header className="frame">
        <div className="registry">
          <Link href="/" className="brand" aria-label="ADA Group — Medical AI Research">
            <Image
              src="/img/logo.png"
              alt="ADA Group"
              width={1264}
              height={843}
              priority
              className="brand-logo"
            />
            <span className="brand-sub">Medical AI · Research</span>
          </Link>
          <div /> {/* spacer */}
          <div className="registry-meta">
            <span className="pulse" />
            <span>6 pipelines online · v1.1</span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main>
        <section className="frame hero">
          <HeroScan />

          <div>
            <p className="hero-eyebrow reveal">
              Sáu pipeline lâm sàng đã triển khai
            </p>
            <h1 className="hero-title reveal" style={{ ['--delay' as string]: '0.05s' }}>
              <em>Định vị</em> tổn thương trong vài giây.
            </h1>
            <p className="hero-sub reveal" style={{ ['--delay' as string]: '0.15s' }}>
              ADA Group dựng pipeline AI cho 6 chuyên khoa,
              chạy trên dataset chuẩn (BraTS, LIDC, CHB-MIT, BUSI) và
              phơi kết quả qua một phòng đọc PACS duy nhất. Mỗi case
              bên dưới là pipeline thật, không phải mockup.
            </p>
            <Link
              href="/legacy.html"
              className="hero-cta reveal"
              style={{ ['--delay' as string]: '0.25s' }}
            >
              Mở phòng đọc demo
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                <path d="M5 12h14M13 6l6 6-6 6" />
              </svg>
            </Link>

            <dl className="hero-spec reveal" style={{ ['--delay' as string]: '0.35s' }}>
              <dt>Stack</dt>
              <dd>Next.js · PyTorch · TF · Three.js</dd>
              <dt>Datasets</dt>
              <dd>BraTS · LIDC · CHB-MIT · BUSI</dd>
              <dt>Modules</dt>
              <dd>EEG · MRI · CT · Blood · Spine · US</dd>
              <dt>3D viz</dt>
              <dd>marching cubes · ellipsoidal taper</dd>
            </dl>
          </div>
        </section>

        <StatsReel />

        <section className="frame cases">
          <div className="cases-header">
            <h2>Sáu case, sáu pipeline, một phòng đọc duy nhất.</h2>
            <p>
              Mỗi case dưới đây là pipeline production đã trained.
              Click để mở demo trực tiếp với dataset thật.
            </p>
          </div>

          <article className="case fx-reveal" data-case="01">
            <div className="case-meta">
              <div className="case-num">CASE 01 · NEURO</div>
              <h3 className="case-title">Phát hiện cơn động kinh từ EEG nhiều kênh</h3>
              <p className="case-desc">
                CNN trích đặc trưng spatial trên 23 channel, BiGRU bắt
                temporal dependency, Attention focus vào segment có
                sóng bất thường. Trained trên CHB-MIT, ROC-AUC 0.84.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>CNN+BiGRU+Att</dd>
                <dt>Filter</dt>      <dd>0.5–40 Hz</dd>
                <dt>Output</dt>      <dd>p(seizure) · t</dd>
              </dl>
              <Link href="/legacy.html#eeg" className="case-link">
                Mở case 01
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <CaseVisualEEG />
          </article>

          <article className="case fx-reveal" data-case="02">
            <div className="case-meta">
              <div className="case-num">CASE 02 · ONCOLOGY</div>
              <h3 className="case-title">Phân đoạn u não đa-mô-thức trên MRI 4 channel</h3>
              <p className="case-desc">
                3D U-Net trained BraTS 2020, input FLAIR + T1 + T1c +
                T2 nguyên kích thước native, predict per-voxel ra ba
                lớp NCR / ED / ET. Mesh 3D extract qua marching cubes
                để render trên brain GLB thật.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>3D U-Net + 4-way TTA</dd>
                <dt>Classes</dt>     <dd>NCR · ED · ET</dd>
                <dt>Render</dt>      <dd>marching cubes</dd>
              </dl>
              <Link href="/legacy.html#brain" className="case-link">
                Mở case 02
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <CaseVisualBrain />
          </article>

          <article className="case fx-reveal" data-case="03">
            <div className="case-meta">
              <div className="case-num">CASE 03 · PULMONOLOGY</div>
              <h3 className="case-title">Định vị nốt phổi & ước lượng malignancy trên CT</h3>
              <p className="case-desc">
                DeepLabV3 segment nốt phổi trên LIDC-IDRI, tính
                đường kính mm theo affine, score malignancy theo
                consensus 4 radiologist. Output 3D phổi GLB với nốt
                được highlight + 3 lát MPR axial/sag/cor đồng bộ.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>DeepLabV3</dd>
                <dt>Range</dt>       <dd>5 – 28 mm</dd>
                <dt>Render</dt>      <dd>GLB + MPR</dd>
              </dl>
              <Link href="/legacy.html#lung" className="case-link">
                Mở case 03
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <CaseVisualLung />
          </article>

          <article className="case fx-reveal" data-case="04">
            <div className="case-meta">
              <div className="case-num">CASE 04 · HEMATOLOGY</div>
              <h3 className="case-title">Đọc bảng xét nghiệm máu, gắn cờ chỉ số bất thường</h3>
              <p className="case-desc">
                Rule-based engine cross-check CBC + Glucose + Lipid
                panel với reference range chuẩn lab, đánh giá nguy cơ
                tim mạch / tiểu đường / thiếu máu, trả về khuyến nghị
                theo dõi. Đơn giản nhưng dùng được ngay.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>Rule-based</dd>
                <dt>Panels</dt>      <dd>CBC · Lipid · Glucose</dd>
                <dt>Output</dt>      <dd>Risk score</dd>
              </dl>
              <Link href="/legacy.html#blood" className="case-link">
                Mở case 04
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <CaseVisualBlood />
          </article>

          <article className="case fx-reveal" data-case="05">
            <div className="case-meta">
              <div className="case-num">CASE 05 · ORTHOPEDICS</div>
              <h3 className="case-title">Tách 24 đốt sống & chỉ ra bệnh lý trên CT cột sống</h3>
              <p className="case-desc">
                TotalSegmentator (nnU-Net trained 1204 case CT toàn thân)
                tách 117 cấu trúc giải phẫu, lọc ra 24 đốt sống C1-L5.
                Mỗi đốt đo riêng chiều cao, đường kính, gắn cờ khi có
                dấu hiệu compression fracture hoặc disc degeneration.
                Chạy trên CT nguyên gốc, không downsample.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>TotalSegmentator (nnU-Net)</dd>
                <dt>Range</dt>       <dd>C1 – L5 · 24 đốt</dd>
                <dt>Output</dt>      <dd>per-vertebra label + flags</dd>
              </dl>
              <Link href="/legacy.html#spine" className="case-link">
                Mở case 05
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <SpineViewer />
          </article>

          <article className="case fx-reveal" data-case="06">
            <div className="case-meta">
              <div className="case-num">CASE 06 · ONCOLOGY · US</div>
              <h3 className="case-title">Khoanh khối u vú trên siêu âm, dựng thể tích 3D từ mask</h3>
              <p className="case-desc">
                BreastCancerSegmentor U-Net 2.16M params trained trên
                BUSI dataset, segment lesion trên ảnh ultrasound 2D.
                Mask được refine bằng DenseCRF, rồi dựng 3D volume bằng
                marching squares contour + ellipsoidal taper. BiomedCLIP
                zero-shot classify ensemble với rule-based BI-RADS từ
                shape features (circularity, solidity, echogenicity).
                Phòng đọc PACS có pan, zoom, ruler, tab 2D ↔ 3D ngay
                trong viewer.
              </p>
              <dl className="case-stats">
                <dt>Method</dt>      <dd>U-Net + BiomedCLIP + DenseCRF</dd>
                <dt>Dataset</dt>     <dd>BUSI · 780 imgs</dd>
                <dt>View</dt>        <dd>PACS workstation + 3D vol</dd>
              </dl>
              <Link href="/legacy.html#breast" className="case-link">
                Mở case 06
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </Link>
            </div>
            <CaseVisualBreast />
          </article>
        </section>

        <ArchitectureFlow />

        <DatasetsCredits />
      </main>

      <footer className="frame footer">
        <div className="footer-grid">
          <div className="footer-brand">
            <div className="footer-logo">
              <Image
                src="/img/logo.png"
                alt="ADA Group"
                width={1264}
                height={843}
                className="brand-logo"
              />
              <div className="footer-tag">
                <span className="brand-sub">Medical AI · Research</span>
              </div>
            </div>
            <p className="footer-mission">
              Sáu pipeline lâm sàng triển khai trong một phòng đọc PACS
              duy nhất. Đứng trên dataset peer-reviewed, viewer chuẩn
              radiologist, mã nguồn mở.
            </p>
          </div>

          <nav className="footer-col" aria-label="Explore cases">
            <div className="footer-col-h">EXPLORE</div>
            <ul>
              <li><Link href="/legacy.html#eeg">Case 01 — EEG</Link></li>
              <li><Link href="/legacy.html#brain">Case 02 — Brain MRI</Link></li>
              <li><Link href="/legacy.html#lung">Case 03 — Lung CT</Link></li>
              <li><Link href="/legacy.html#blood">Case 04 — Blood</Link></li>
              <li><Link href="/legacy.html#spine">Case 05 — Spine CT</Link></li>
              <li><Link href="/legacy.html#breast">Case 06 — Breast US</Link></li>
            </ul>
          </nav>

          <div className="footer-col">
            <div className="footer-col-h">BUILT WITH</div>
            <ul className="footer-stack">
              <li>Next.js 16 · React 19</li>
              <li>Python Flask · TensorFlow</li>
              <li>PyTorch · Ultralytics YOLO</li>
              <li>Three.js · OrbitControls</li>
              <li>nnU-Net · BiomedCLIP</li>
              <li>DenseCRF · marching cubes</li>
            </ul>
          </div>

          <div className="footer-col">
            <div className="footer-col-h">CONTACT</div>
            <ul className="footer-contact">
              <li>
                <span className="footer-k">eng</span>
                <a href="mailto:aihoclaptrinh@gmail.com">TruongTanNghia</a>
              </li>
              <li>
                <span className="footer-k">repo</span>
                <a href="https://github.com/TruongTanNghia/Project_Em_Dat" target="_blank" rel="noopener noreferrer">
                  github · Project_Em_Dat
                </a>
              </li>
              <li>
                <span className="footer-k">demo</span>
                <Link href="/legacy.html">Mở phòng đọc</Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="footer-bar">
          <span className="footer-copy">© 2026 ADA Group · TruongTanNghia · v1.1</span>
          <span className="footer-warn">
            Demo cho research và educational use.
            Không dùng cho chẩn đoán lâm sàng — luôn cần bác sĩ board-certified xem lại.
          </span>
        </div>
      </footer>
    </div>
  );
}
