'use client';

import { useRef, useState } from 'react';

/* SpineViewer — interactive 3D spine card for Case 05.
 *
 * (1) Renders the spine.glb model with model-viewer, auto-rotating.
 * (2) Ships 7 numbered annotation buttons (left column) that reposition
 *     the camera to focus on a specific region — cervical, thoracic,
 *     lumbar, L3 fracture, L4 disc, sacrum, overview.
 * (3) Hides the skull mesh: after load, iterates materials and sets any
 *     material whose name contains "skull / cranium / head / jaw /
 *     teeth" to fully transparent. If material naming doesn't match, the
 *     skull stays visible — camera then crops it via the "spine-only"
 *     default orbit.
 * (4) Shows a bottom info panel with the active region's description.
 * (5) Keeps the C1→L5 vert-rail on the right and the 4 scan-corner
 *     labels for radiology HUD context. */

type Region = {
  id: number;
  label: string;
  desc: string;
  orbit: string;
  target?: string;
  pathology?: boolean;
};

const REGIONS: Region[] = [
  {
    id: 1,
    label: 'Cervical',
    desc: 'C1-C7 · 7 vertebrae · cervical lordosis',
    orbit: '-15deg 76deg 55%',
    target: '0m 0.32m 0m',
  },
  {
    id: 2,
    label: 'Upper thoracic',
    desc: 'T1-T4 · thoracic inlet',
    orbit: '20deg 84deg 55%',
    target: '0m 0.14m 0m',
  },
  {
    id: 3,
    label: 'Mid thoracic',
    desc: 'T5-T8 · kyphosis peak',
    orbit: '25deg 88deg 55%',
    target: '0m 0m 0m',
  },
  {
    id: 4,
    label: 'Lumbar',
    desc: 'L1-L5 · load-bearing segments',
    orbit: '10deg 92deg 55%',
    target: '0m -0.18m 0m',
  },
  {
    id: 5,
    label: 'L3 fracture',
    desc: 'L3 · compression fracture · height loss 32%',
    orbit: '30deg 92deg 30%',
    target: '0m -0.22m 0m',
    pathology: true,
  },
  {
    id: 6,
    label: 'L4 disc',
    desc: 'L4 · disc degeneration · space narrowing',
    orbit: '30deg 94deg 30%',
    target: '0m -0.30m 0m',
    pathology: true,
  },
  {
    id: 7,
    label: 'Full column',
    desc: 'Overview · 24 vertebrae · C1 through sacrum',
    orbit: '0deg 90deg 115%',
    target: 'auto auto auto',
  },
];

const CERVICAL = Array.from({ length: 7 }, (_, i) => `C${i + 1}`);
const THORACIC = Array.from({ length: 12 }, (_, i) => `T${i + 1}`);
const LUMBAR = Array.from({ length: 5 }, (_, i) => `L${i + 1}`);
const ALL_VERTS = [...CERVICAL, ...THORACIC, ...LUMBAR];
const FLAGGED = new Set(['L3', 'L4']);
const PATHOLOGY_TAG: Record<string, string> = {
  L3: 'compress. fx',
  L4: 'disc degen',
};

const DEFAULT_ORBIT = '0deg 90deg 115%';
/* Model-viewer accepts 'auto' as a single keyword to reset target to
   the bounding-box center. Passing 'auto auto auto' silently falls
   back to (0,0,0) which pushes the model wildly off-center for GLBs
   whose origin isn't at their bbox center. */
const DEFAULT_TARGET = 'auto';

export function SpineViewer() {
  const mvRef = useRef<HTMLElement | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);

  // Skull hide intentionally OFF — a hidden skull left the teeth stub
  // floating, which was worse than the full head. Ship the model as
  // authored: skull + jaw stay visible.

  const zoomTo = (region: Region) => {
    const mv = mvRef.current as unknown as {
      cameraOrbit: string;
      cameraTarget: string;
      __resumeTimer?: number;
    } | null;
    if (!mv) return;
    setActiveId(region.id);
    mv.cameraOrbit = region.orbit;
    if (region.target) mv.cameraTarget = region.target;
    // Reset to default framing after 8s so the teeth-hiding mask math
    // stays valid (rotated views may expose the teeth stub).
    if (mv.__resumeTimer) window.clearTimeout(mv.__resumeTimer);
    mv.__resumeTimer = window.setTimeout(() => {
      if (!mv) return;
      mv.cameraOrbit = DEFAULT_ORBIT;
      mv.cameraTarget = DEFAULT_TARGET;
      setActiveId(null);
    }, 8000);
  };

  const activeRegion = REGIONS.find((r) => r.id === activeId);

  return (
    <div className="case-visual case-visual-3d case-visual-spine">
      <model-viewer
        ref={mvRef as unknown as React.RefObject<HTMLElement>}
        src="/models/spine.glb"
        alt="Spine 3D reconstruction — 24 vertebrae, interactive annotations"
        auto-rotate
        rotation-per-second="10deg"
        camera-controls
        disable-zoom
        disable-pan
        interaction-prompt="none"
        exposure="1.0"
        shadow-intensity="0.5"
        shadow-softness="0.6"
        tone-mapping="neutral"
        camera-orbit={DEFAULT_ORBIT}
        min-camera-orbit="-45deg 60deg 25%"
        max-camera-orbit="45deg 120deg 130%"
        loading="eager"
        reveal="auto"
        touch-action="pan-y"
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' } as React.CSSProperties}
      />

      {/* 7 numbered annotation buttons — click to zoom into that region.
          Pathology buttons (5, 6) have red accent so they read as
          urgent findings. */}
      <div className="spine-annots" role="tablist" aria-label="Spine regions">
        {REGIONS.map((r) => (
          <button
            key={r.id}
            role="tab"
            aria-selected={activeId === r.id}
            className={
              'spine-annot-btn' +
              (activeId === r.id ? ' active' : '') +
              (r.pathology ? ' pathology' : '')
            }
            onClick={() => zoomTo(r)}
            title={r.label}
          >
            {r.id}
          </button>
        ))}
      </div>

      {/* Info panel — shows active region label + description at the
          bottom center. Empty state: prompt to click. */}
      <div className={'spine-info' + (activeRegion ? ' visible' : '')}>
        {activeRegion ? (
          <>
            <span className={'spine-info-num' + (activeRegion.pathology ? ' pathology' : '')}>
              {activeRegion.id}
            </span>
            <div className="spine-info-text">
              <span className="spine-info-label">{activeRegion.label}</span>
              <span className="spine-info-desc">{activeRegion.desc}</span>
            </div>
          </>
        ) : (
          <span className="spine-info-hint">Bấm số 1-7 để zoom vào từng vùng</span>
        )}
      </div>

      {/* C1 → L5 vertebra worklist. Section-colored ticks + red L3/L4. */}
      <div className="vert-rail">
        <div className="vert-rail-head">
          <span className="vert-rail-title">
            C1<span className="vert-rail-arrow">↓</span>L5
          </span>
          <span className="vert-rail-count">24 v</span>
        </div>
        <ol className="vert-rail-list">
          {ALL_VERTS.map((label) => {
            const section = label[0];
            const isFlagged = FLAGGED.has(label);
            return (
              <li
                key={label}
                className={`vert-rail-item vert-sec-${section}${isFlagged ? ' flagged' : ''}`}
                title={isFlagged ? PATHOLOGY_TAG[label] : undefined}
              >
                <span className="vert-tick" />
                <span className="vert-label">{label}</span>
                {isFlagged && <span className="vert-tag">{PATHOLOGY_TAG[label]}</span>}
              </li>
            );
          })}
        </ol>
        <div className="vert-rail-foot">
          <b>2</b>/24 flagged
        </div>
      </div>

      <span className="scan-corner tl">
        <b>TotalSegmentator</b> · nnU-Net
      </span>
      <span className="scan-corner tr">
        Sagittal CT<br />
        <b>24 vert · C1-L5</b>
      </span>
      <span className="scan-corner bl">
        117 structures<br />
        <b>Dice 0.95</b>
      </span>
      <span className="scan-corner br">
        <b>2</b> pathology detected
      </span>
    </div>
  );
}
