/* ──────────────────────────────────────────────────────────────
   anatomy.js — section "MÔ PHỎNG GIẢI PHẨU 3D" cho U não MRI tab.

   Tận dụng Three.js + OrbitControls + GLTFLoader đã có sẵn trong
   legacy.html. Self-contained IIFE, lazy-init khi section vào viewport.

   APIs đối ngoại:
   - Lắng nghe custom event `brain:tumor-detected`
       detail: { location, volume, coords?, side? }
   - Cho phép gọi thủ công: window.AnatomyView.showTumor({...})
─────────────────────────────────────────────────────────────── */

(function () {
  'use strict';

  /* ─── Lobe definitions ──────────────────────────────────
     Position: trong toạ độ model space của human-brain.glb.
     CameraPos: vị trí camera khi user click lobe đó (camera luôn
     look at origin).
  */
  const LOBES = [
    {
      id: 'frontal',
      name: 'Thuỳ trán',
      position: [0, 0.030, 0.062],
      cameraPos: [0, 0.04, 0.20],
      aiFocus: false,
    },
    {
      id: 'parietal',
      name: 'Thuỳ đỉnh',
      position: [0, 0.062, 0],
      cameraPos: [0, 0.20, 0.06],
      aiFocus: false,
    },
    {
      id: 'temporal',
      name: 'Thuỳ thái dương',
      position: [0.058, 0.005, 0.040],
      cameraPos: [0.20, 0.02, 0.08],
      aiFocus: true,
    },
    {
      id: 'occipital',
      name: 'Thuỳ chẩm',
      position: [0, 0.020, -0.062],
      cameraPos: [0, 0.04, -0.20],
      aiFocus: false,
    },
    {
      id: 'cerebellum',
      name: 'Tiểu não',
      position: [0, -0.040, -0.035],
      cameraPos: [0, -0.14, -0.14],
      aiFocus: false,
    },
    {
      id: 'brainstem',
      name: 'Thân não',
      position: [0, -0.058, -0.005],
      cameraPos: [0, -0.20, 0.06],
      aiFocus: false,
    },
  ];

  const DEFAULT_CAMERA = [0, 0.04, 0.25];

  /* ─── State ─────────────────────────────────────────────── */
  let scene, camera, renderer, controls;
  let brainModel = null;
  let hotspots = []; // { mesh, ring, lobe }
  let tumorSphere = null;
  let raycaster, mouse;
  let canvas, container;
  let cameraTween = null;
  let autoRotate = true;
  let initialized = false;

  // Mô phỏng cắt lớp — clipping plane state
  const slice = {
    mode: 'off',          // 'off' | 'axial' | 'sagittal' | 'coronal'
    pos:  0,              // -1 ... 1 (normalized within ±BRAIN_HALF)
    plane: new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),
    markerMesh: null,
  };
  const BRAIN_HALF = 0.07;  // half of normalized brain (matches TARGET_MAX/2 in loadBrain)

  /* ─── Lazy init ─────────────────────────────────────────── */
  function bootstrap() {
    canvas = document.getElementById('anatomyCanvas');
    container = document.getElementById('anatomyViewer');
    if (!canvas) return;
    if (typeof IntersectionObserver === 'undefined') {
      init();
      return;
    }
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          init();
          io.disconnect();
          return;
        }
      }
    }, { rootMargin: '300px 0px' });
    io.observe(canvas);
  }

  function init() {
    if (initialized) return;
    if (typeof THREE === 'undefined') {
      // Three.js chưa load xong → retry
      setTimeout(init, 250);
      return;
    }
    if (typeof THREE.GLTFLoader === 'undefined') {
      setTimeout(init, 250);
      return;
    }

    setupScene();
    loadBrain();
    setupRaycaster();
    setupAtlasCards();
    setupResetButton();
    setupSliceControls();
    setupStageNav();
    listenForTumorEvent();
    window.addEventListener('resize', onResize, { passive: true });
    animate();
    initialized = true;
  }

  /* ─── Scene / camera / lights ───────────────────────────── */
  function setupScene() {
    const rect = container.getBoundingClientRect();
    const W = rect.width || 600;
    const H = rect.height || 460;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(38, W / H, 0.005, 50);
    camera.position.fromArray(DEFAULT_CAMERA);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H, false);
    renderer.localClippingEnabled = true;  // for "mô phỏng cắt lớp"

    // Lighting — warm neutral, no heavy blue tint
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));

    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1, 1.2, 0.8);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xfff0e6, 0.35);
    fill.position.set(-1, -0.4, 0.5);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0xb19cff, 0.22);
    rim.position.set(0, 0.2, -1);
    scene.add(rim);

    // OrbitControls — let user drag, no zoom/pan
    if (typeof THREE.OrbitControls !== 'undefined') {
      controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enableZoom = false;
      controls.enablePan = false;
      controls.rotateSpeed = 0.6;
      controls.minPolarAngle = Math.PI * 0.18;
      controls.maxPolarAngle = Math.PI * 0.85;
      controls.addEventListener('start', () => {
        autoRotate = false;
        cameraTween = null;
      });
    }
  }

  /* ─── Load brain GLB ────────────────────────────────────── */
  function loadBrain() {
    const loader = new THREE.GLTFLoader();
    loader.load(
      '/models/brain/human-brain.glb',
      (gltf) => {
        brainModel = gltf.scene;

        // 1) Compute native bbox
        let box = new THREE.Box3().setFromObject(brainModel);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;

        // 2) Normalize scale: max dim = 0.14 (so hotspot coords [±0.06]
        //    sit just inside the surface). GLB native could be 1 unit
        //    or 100; this makes the scene predictable.
        const TARGET_MAX = 0.14;
        const scale = TARGET_MAX / maxDim;
        brainModel.scale.setScalar(scale);

        // 3) Recompute bbox after scaling, then center at origin
        box = new THREE.Box3().setFromObject(brainModel);
        const center = box.getCenter(new THREE.Vector3());
        brainModel.position.sub(center);

        // 4) Override material: GLB's native color is saturated cyan
        //    which doesn't read as brain tissue. Replace with a warm
        //    pinkish-grey Phong material so it looks like real mô não.
        brainModel.traverse((node) => {
          if (node.isMesh) {
            const newMat = new THREE.MeshPhongMaterial({
              color:        0xd9b0a4,  // pinkish flesh
              specular:     0xffe6dc,
              shininess:    18,
              transparent:  true,
              opacity:      0.78,
              depthWrite:   false,
              side:         THREE.DoubleSide,
              clippingPlanes: [],
              clipShadows:  true,
            });
            node.material = newMat;
            node.userData.brainMesh = true;
          }
        });

        scene.add(brainModel);
        addHotspots();
        hidePlaceholder();
      },
      undefined,
      (err) => {
        console.error('[anatomy] Failed to load brain GLB:', err);
        showError('Không tải được mô hình não. Kiểm tra mạng?');
      }
    );
  }

  /* ─── Hotspots (clickable spheres at lobe centres) ──────── */
  function addHotspots() {
    LOBES.forEach((lobe) => {
      const color = lobe.aiFocus ? 0xff6b5a : 0xa78bfa;

      const dotGeom = new THREE.SphereGeometry(0.005, 18, 18);
      const dotMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const dot = new THREE.Mesh(dotGeom, dotMat);
      dot.position.fromArray(lobe.position);
      dot.userData.lobe = lobe.id;
      dot.renderOrder = 999;
      scene.add(dot);

      const ringGeom = new THREE.SphereGeometry(0.0085, 18, 18);
      const ringMat = new THREE.MeshBasicMaterial({
        color: color,
        transparent: true,
        opacity: 0.25,
        depthTest: false,
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.fromArray(lobe.position);
      ring.renderOrder = 998;
      scene.add(ring);

      hotspots.push({ mesh: dot, ring: ring, lobe: lobe });
    });
  }

  /* ─── Raycaster — click hotspot in canvas ──────────────── */
  function setupRaycaster() {
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    canvas.addEventListener('click', onCanvasClick);
  }

  function onCanvasClick(evt) {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hitMeshes = hotspots.map((h) => h.mesh);
    const hits = raycaster.intersectObjects(hitMeshes);
    if (hits.length > 0) {
      selectLobe(hits[0].object.userData.lobe);
    }
  }

  /* ─── Atlas card click handlers ────────────────────────── */
  function setupAtlasCards() {
    document.querySelectorAll('[data-anatomy-lobe]').forEach((card) => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-anatomy-lobe');
        selectLobe(id);
      });
    });
  }

  function setupResetButton() {
    const btn = document.getElementById('anatomyReset');
    if (!btn) return;
    btn.addEventListener('click', () => {
      autoRotate = true;
      tweenCameraTo(DEFAULT_CAMERA, 800);
      document.querySelectorAll('[data-anatomy-lobe]').forEach((c) => {
        c.classList.remove('is-active');
      });
    });
  }

  /* ─── Selecting a lobe → animate camera + highlight ───── */
  function selectLobe(lobeId) {
    const lobe = LOBES.find((l) => l.id === lobeId);
    if (!lobe) return;

    // Sync atlas UI
    document.querySelectorAll('[data-anatomy-lobe]').forEach((c) => {
      c.classList.toggle('is-active', c.getAttribute('data-anatomy-lobe') === lobeId);
    });

    autoRotate = false;
    tweenCameraTo(lobe.cameraPos, 900);

    const h = hotspots.find((x) => x.lobe.id === lobeId);
    if (h) pulseHotspotOnce(h);
  }

  function tweenCameraTo(targetArr, duration) {
    cameraTween = {
      from: camera.position.clone(),
      to: new THREE.Vector3().fromArray(targetArr),
      startedAt: performance.now(),
      duration: duration || 900,
    };
  }

  function pulseHotspotOnce(h) {
    const start = performance.now();
    function tick(now) {
      const t = (now - start) / 800;
      if (t > 1) {
        h.ring.scale.set(1, 1, 1);
        h.ring.material.opacity = 0.25;
        return;
      }
      const scale = 1 + Math.sin(t * Math.PI) * 1.7;
      h.ring.scale.set(scale, scale, scale);
      h.ring.material.opacity = 0.25 + (1 - t) * 0.55;
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ─── Tumor overlay ────────────────────────────────────── */
  function listenForTumorEvent() {
    document.addEventListener('brain:tumor-detected', (e) => {
      const data = e.detail || {};
      showTumor(data);
    });
  }

  function showTumor(data) {
    if (tumorSphere) {
      scene.remove(tumorSphere);
      tumorSphere.geometry.dispose();
      tumorSphere.material.dispose();
      tumorSphere = null;
    }

    const pos = data.position || [0.048, 0.005, 0.035]; // default = temporal R
    const volumeCm3 = (typeof data.volume === 'number') ? data.volume : 6.0;

    // Radius scaled from cm³ volume: V = 4/3 π r³ → r = (3V/4π)^(1/3) (cm)
    // Convert cm → model units (rough ~ ÷ 100), clamp for sanity
    const realRcm = Math.cbrt((3 * volumeCm3) / (4 * Math.PI));
    const radius = Math.max(0.008, Math.min(0.022, realRcm / 100 * 0.45));

    const geom = new THREE.SphereGeometry(radius, 28, 28);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xff4d6d,
      transparent: true,
      opacity: 0.78,
      depthTest: false,
    });
    tumorSphere = new THREE.Mesh(geom, mat);
    tumorSphere.position.fromArray(pos);
    tumorSphere.renderOrder = 1000;
    scene.add(tumorSphere);

    // Glow ring around tumor
    const glowGeom = new THREE.SphereGeometry(radius * 1.8, 24, 24);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0xff4d6d,
      transparent: true,
      opacity: 0.18,
      depthTest: false,
    });
    const glow = new THREE.Mesh(glowGeom, glowMat);
    glow.position.fromArray(pos);
    glow.renderOrder = 999;
    scene.add(glow);
    tumorSphere.userData.glow = glow;

    updateTumorPanel(data, pos);
  }

  function updateTumorPanel(data, pos) {
    const panel = document.getElementById('anatomyTumorInfo');
    if (!panel) return;
    panel.hidden = false;

    const locEl = document.getElementById('anatomyTumorLocation');
    const volEl = document.getElementById('anatomyTumorVolume');
    const coordEl = document.getElementById('anatomyTumorCoords');

    if (locEl) locEl.textContent = data.location || 'Temporal lobe · R';
    if (volEl) {
      const v = (typeof data.volume === 'number') ? data.volume : 6.0;
      volEl.textContent = v.toFixed(1) + ' cm³';
    }
    if (coordEl) {
      coordEl.textContent =
        `X=${(pos[0] * 1000).toFixed(0)} ` +
        `Y=${(pos[1] * 1000).toFixed(0)} ` +
        `Z=${(pos[2] * 1000).toFixed(0)} mm`;
    }
  }

  /* ─── Resize ────────────────────────────────────────────── */
  function onResize() {
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const W = rect.width || 600;
    const H = rect.height || 460;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H, false);
  }

  /* ─── Animate loop ──────────────────────────────────────── */
  function animate() {
    requestAnimationFrame(animate);

    // Auto-rotate the brain when user isn't dragging
    if (brainModel && autoRotate && !cameraTween) {
      brainModel.rotation.y += 0.0025;
    }

    // Camera tween (when user clicked an atlas item)
    if (cameraTween) {
      const t = Math.min(1, (performance.now() - cameraTween.startedAt) / cameraTween.duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      camera.position.lerpVectors(cameraTween.from, cameraTween.to, eased);
      camera.lookAt(0, 0, 0);
      if (t >= 1) cameraTween = null;
    }

    if (controls) controls.update();

    // Subtle continuous pulse on the AI-focus hotspot ring
    const focus = hotspots.find((h) => h.lobe.aiFocus);
    if (focus) {
      const phase = (performance.now() / 1400) % (Math.PI * 2);
      const s = 1 + Math.sin(phase) * 0.18;
      focus.ring.scale.set(s, s, s);
    }

    renderer.render(scene, camera);
  }

  /* ─── Loader UI ─────────────────────────────────────────── */
  function hidePlaceholder() {
    const ph = document.getElementById('anatomyPlaceholder');
    if (!ph) return;
    ph.style.opacity = '0';
    setTimeout(() => { ph.style.display = 'none'; }, 400);
  }

  function showError(msg) {
    const ph = document.getElementById('anatomyPlaceholder');
    if (!ph) return;
    ph.innerHTML = '<span style="color:#ef4444; max-width: 220px; text-align: center;">' +
                   msg + '</span>';
  }

  /* ─── Mô phỏng cắt lớp (clipping plane simulation) ──────── */
  function setupSliceControls() {
    // Visual marker — a translucent purple quad showing where the cut is
    const geom = new THREE.PlaneGeometry(0.22, 0.22);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xa78bfa,
      transparent: true,
      opacity: 0.20,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    slice.markerMesh = new THREE.Mesh(geom, mat);
    slice.markerMesh.visible = false;
    slice.markerMesh.renderOrder = 500;
    scene.add(slice.markerMesh);

    // Button group
    document.querySelectorAll('[data-slice-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-slice-mode');
        setSliceMode(mode);
      });
    });

    // Slider
    const slider = document.getElementById('anatomySliceSlider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        slice.pos = parseFloat(e.target.value);
        updateSlicePlane();
        updateSliceReadout();
      });
    }
  }

  function setSliceMode(mode) {
    slice.mode = mode;
    slice.pos = 0;
    const slider = document.getElementById('anatomySliceSlider');
    if (slider) {
      slider.value = '0';
      slider.disabled = (mode === 'off');
    }

    document.querySelectorAll('[data-slice-mode]').forEach((b) => {
      b.classList.toggle('is-active', b.getAttribute('data-slice-mode') === mode);
    });

    if (mode === 'off') {
      if (slice.markerMesh) slice.markerMesh.visible = false;
      applyClippingPlanes([]);
      updateSliceReadout();
      return;
    }

    updateSlicePlane();
    if (slice.markerMesh) slice.markerMesh.visible = true;
    applyClippingPlanes([slice.plane]);
    updateSliceReadout();
  }

  function updateSlicePlane() {
    if (!slice.markerMesh) return;
    const pos = slice.pos * BRAIN_HALF;

    switch (slice.mode) {
      case 'axial':    // horizontal cut, normal pointing down (+Y above kept, below clipped)
        slice.plane.normal.set(0, -1, 0);
        slice.plane.constant = pos;
        slice.markerMesh.position.set(0, pos, 0);
        slice.markerMesh.rotation.set(-Math.PI / 2, 0, 0);
        break;
      case 'sagittal': // vertical cut left-right
        slice.plane.normal.set(-1, 0, 0);
        slice.plane.constant = pos;
        slice.markerMesh.position.set(pos, 0, 0);
        slice.markerMesh.rotation.set(0, Math.PI / 2, 0);
        break;
      case 'coronal':  // front-back cut
        slice.plane.normal.set(0, 0, -1);
        slice.plane.constant = pos;
        slice.markerMesh.position.set(0, 0, pos);
        slice.markerMesh.rotation.set(0, 0, 0);
        break;
    }
  }

  function applyClippingPlanes(planes) {
    if (!brainModel) return;
    brainModel.traverse((node) => {
      if (node.userData.brainMesh && node.material) {
        node.material.clippingPlanes = planes;
      }
    });
  }

  function updateSliceReadout() {
    const out = document.getElementById('anatomySliceReadout');
    if (!out) return;
    if (slice.mode === 'off') {
      out.textContent = '—';
      return;
    }
    const mm = Math.round(slice.pos * 70); // brain ≈ 14cm → ±70mm
    const labels = { axial: 'Z', sagittal: 'X', coronal: 'Y' };
    out.textContent = `${slice.mode.toUpperCase()} · ${labels[slice.mode]} = ${mm > 0 ? '+' : ''}${mm} mm`;
  }

  /* ─── Stage navigation (4-stage workflow) ──────────────── */
  function setupStageNav() {
    const buttons = document.querySelectorAll('[data-stage]');
    const panels  = document.querySelectorAll('[data-stage-panel]');
    if (!buttons.length || !panels.length) return;

    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-stage');

        buttons.forEach((b) => {
          b.classList.toggle('is-active', b === btn);
        });

        panels.forEach((p) => {
          const isActive = p.getAttribute('data-stage-panel') === target;
          p.classList.toggle('is-active', isActive);
          p.hidden = !isActive;
        });

        // Trigger a resize so the WebGL canvas re-fits if we just
        // switched back to the model stage (canvas was display:none)
        if (target === 'model') {
          requestAnimationFrame(onResize);
        }

        // Lazy-init Stage 2 on first visit
        if (target === 'plan' && !planState.initialized) {
          requestAnimationFrame(initPlanStage);
        } else if (target === 'plan') {
          requestAnimationFrame(planOnResize);
        }
      });
    });
  }

  /* ═══════════════════════════════════════════════════════════
     STAGE 2 — Surgical Path Planning
     Riêng scene + brain + probe + zones avoid metrics.
  ═══════════════════════════════════════════════════════════ */

  const TUMOR_TARGET = new THREE.Vector3(0.045, 0.005, 0.035); // Temporal R region
  const PROBE_LENGTH = 0.13; // model units, entry above skull surface

  // Vessel zones — green halos to avoid
  const VESSEL_ZONES = [
    { pos: [ 0.000, -0.045,  0.000], r: 0.018, name: 'Circle of Willis' },
    { pos: [ 0.045, -0.012,  0.028], r: 0.012, name: 'MCA R trunk' },
    { pos: [-0.045, -0.012,  0.028], r: 0.012, name: 'MCA L trunk' },
    { pos: [ 0.000,  0.058,  0.000], r: 0.012, name: 'Sagittal sinus' },
    { pos: [ 0.030, -0.020, -0.035], r: 0.010, name: 'PCA R' },
  ];

  // Functional zones — purple halos to avoid
  const FUNC_ZONES = [
    { pos: [ 0.032,  0.028,  0.038], r: 0.012, name: 'Motor cortex R' },
    { pos: [-0.032,  0.028,  0.038], r: 0.012, name: 'Motor cortex L' },
    { pos: [-0.046,  0.005,  0.038], r: 0.012, name: 'Broca area L' },
    { pos: [-0.052, -0.002, -0.012], r: 0.013, name: 'Wernicke area L' },
    { pos: [ 0.000,  0.030, -0.055], r: 0.014, name: 'Visual cortex' },
  ];

  const planState = {
    scene: null, camera: null, renderer: null, controls: null,
    brainModel: null,
    canvas: null, container: null,
    probeMesh: null, probeArrow: null,
    tumorSphere: null,
    vesselMeshes: [],
    funcMeshes: [],
    visibility: { probe: true, vessels: true, func: true },
    azimuth: Math.PI,         // 180°
    elevation: Math.PI / 6,   // 30°
    initialized: false,
    rafId: null,
  };

  function initPlanStage() {
    if (planState.initialized) return;
    if (typeof THREE === 'undefined') {
      setTimeout(initPlanStage, 250);
      return;
    }
    if (typeof THREE.GLTFLoader === 'undefined') {
      setTimeout(initPlanStage, 250);
      return;
    }
    planState.canvas = document.getElementById('planCanvas');
    planState.container = document.getElementById('planViewer');
    if (!planState.canvas || !planState.container) return;

    setupPlanScene();
    loadPlanBrain();
    addTumor();
    addVesselZones();
    addFuncZones();
    buildProbe();
    setupPlanControls();
    window.addEventListener('resize', planOnResize, { passive: true });
    animatePlan();
    planState.initialized = true;
  }

  function setupPlanScene() {
    const rect = planState.container.getBoundingClientRect();
    const W = rect.width || 600, H = rect.height || 460;

    planState.scene = new THREE.Scene();
    planState.camera = new THREE.PerspectiveCamera(40, W / H, 0.005, 50);
    planState.camera.position.set(0.06, 0.08, 0.25);
    planState.camera.lookAt(0, 0, 0);

    planState.renderer = new THREE.WebGLRenderer({
      canvas: planState.canvas, antialias: true, alpha: true,
    });
    planState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    planState.renderer.setSize(W, H, false);

    planState.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const k = new THREE.DirectionalLight(0xffffff, 0.85);
    k.position.set(1, 1.2, 0.8); planState.scene.add(k);
    const f = new THREE.DirectionalLight(0xfff0e6, 0.30);
    f.position.set(-1, -0.4, 0.5); planState.scene.add(f);
    const r = new THREE.DirectionalLight(0xb19cff, 0.20);
    r.position.set(0, 0.2, -1); planState.scene.add(r);

    if (typeof THREE.OrbitControls !== 'undefined') {
      planState.controls = new THREE.OrbitControls(planState.camera, planState.canvas);
      planState.controls.enableDamping = true;
      planState.controls.dampingFactor = 0.08;
      planState.controls.enableZoom = false;
      planState.controls.enablePan = false;
      planState.controls.rotateSpeed = 0.6;
      planState.controls.minPolarAngle = Math.PI * 0.15;
      planState.controls.maxPolarAngle = Math.PI * 0.85;
    }
  }

  function loadPlanBrain() {
    const loader = new THREE.GLTFLoader();
    loader.load('/models/brain/human-brain.glb', (gltf) => {
      const m = gltf.scene;
      let box = new THREE.Box3().setFromObject(m);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      m.scale.setScalar(0.14 / maxDim);
      box = new THREE.Box3().setFromObject(m);
      m.position.sub(box.getCenter(new THREE.Vector3()));

      m.traverse((n) => {
        if (n.isMesh) {
          n.material = new THREE.MeshPhongMaterial({
            color: 0xd9b0a4, specular: 0xffe6dc, shininess: 18,
            transparent: true, opacity: 0.55,
            depthWrite: false, side: THREE.DoubleSide,
          });
        }
      });
      planState.brainModel = m;
      planState.scene.add(m);
      hidePlanPlaceholder();
    }, undefined, (err) => {
      console.error('[plan] brain load failed', err);
    });
  }

  function addTumor() {
    const geom = new THREE.SphereGeometry(0.011, 24, 24);
    const mat = new THREE.MeshPhongMaterial({
      color: 0xef4444, emissive: 0x991010, emissiveIntensity: 0.4,
      shininess: 30, transparent: true, opacity: 0.92, depthWrite: false,
    });
    planState.tumorSphere = new THREE.Mesh(geom, mat);
    planState.tumorSphere.position.copy(TUMOR_TARGET);
    planState.tumorSphere.renderOrder = 700;
    planState.scene.add(planState.tumorSphere);
  }

  function addVesselZones() {
    VESSEL_ZONES.forEach((z) => {
      const geom = new THREE.SphereGeometry(z.r, 20, 20);
      const mat = new THREE.MeshBasicMaterial({
        color: 0x10b981, transparent: true, opacity: 0.18,
        depthWrite: false,
      });
      const sphere = new THREE.Mesh(geom, mat);
      sphere.position.fromArray(z.pos);
      sphere.renderOrder = 600;
      planState.scene.add(sphere);
      planState.vesselMeshes.push({
        mesh: sphere,
        pos: new THREE.Vector3().fromArray(z.pos),
        r: z.r, name: z.name,
      });
    });
  }

  function addFuncZones() {
    FUNC_ZONES.forEach((z) => {
      const geom = new THREE.SphereGeometry(z.r, 20, 20);
      const mat = new THREE.MeshBasicMaterial({
        color: 0xc084fc, transparent: true, opacity: 0.18,
        depthWrite: false,
      });
      const sphere = new THREE.Mesh(geom, mat);
      sphere.position.fromArray(z.pos);
      sphere.renderOrder = 600;
      planState.scene.add(sphere);
      planState.funcMeshes.push({
        mesh: sphere,
        pos: new THREE.Vector3().fromArray(z.pos),
        r: z.r, name: z.name,
      });
    });
  }

  function probeDirection() {
    const az = planState.azimuth, el = planState.elevation;
    return new THREE.Vector3(
      Math.sin(az) * Math.cos(el),
      Math.sin(el),
      Math.cos(az) * Math.cos(el)
    ).normalize();
  }

  function probeEntry() {
    return TUMOR_TARGET.clone().add(probeDirection().multiplyScalar(PROBE_LENGTH));
  }

  function buildProbe() {
    if (planState.probeMesh) {
      planState.scene.remove(planState.probeMesh);
      planState.probeMesh.geometry.dispose();
      planState.probeMesh.material.dispose();
    }
    if (planState.probeArrow) {
      planState.scene.remove(planState.probeArrow);
      planState.probeArrow.geometry.dispose();
      planState.probeArrow.material.dispose();
    }

    const entry = probeEntry();
    const target = TUMOR_TARGET;

    // Yellow tube from entry to tumor
    const curve = new THREE.LineCurve3(entry, target);
    const tubeGeom = new THREE.TubeGeometry(curve, 4, 0.0024, 14, false);
    const tubeMat = new THREE.MeshPhongMaterial({
      color: 0xfacc15, emissive: 0xfacc15, emissiveIntensity: 0.5,
      shininess: 60, transparent: true, opacity: 0.95, depthWrite: false,
    });
    planState.probeMesh = new THREE.Mesh(tubeGeom, tubeMat);
    planState.probeMesh.renderOrder = 800;
    planState.scene.add(planState.probeMesh);

    // Cone tip at entry side
    const coneGeom = new THREE.ConeGeometry(0.006, 0.012, 16);
    coneGeom.translate(0, 0.006, 0);
    coneGeom.rotateX(Math.PI / 2);
    const coneMat = new THREE.MeshPhongMaterial({
      color: 0xfde047, emissive: 0xfacc15, emissiveIntensity: 0.4,
      transparent: true, opacity: 0.95, depthWrite: false,
    });
    planState.probeArrow = new THREE.Mesh(coneGeom, coneMat);
    planState.probeArrow.position.copy(entry);
    const lookDir = target.clone().sub(entry).normalize();
    planState.probeArrow.lookAt(entry.clone().add(lookDir));
    planState.probeArrow.renderOrder = 800;
    planState.scene.add(planState.probeArrow);

    updateMetrics();
  }

  function distancePointToSegment(p, a, b) {
    const ab = b.clone().sub(a);
    const ap = p.clone().sub(a);
    const denom = ab.lengthSq();
    if (denom === 0) return p.distanceTo(a);
    const t = Math.max(0, Math.min(1, ap.dot(ab) / denom));
    const proj = a.clone().add(ab.multiplyScalar(t));
    return p.distanceTo(proj);
  }

  function updateMetrics() {
    const entry = probeEntry();
    const target = TUMOR_TARGET;

    // Probe length in mm (model unit 0.14 ≈ 14cm brain)
    const lenMm = entry.distanceTo(target) * 1000;

    // Nearest vessel
    let nearV = { dist: Infinity, name: '—' };
    planState.vesselMeshes.forEach((v) => {
      const d = Math.max(0, distancePointToSegment(v.pos, entry, target) - v.r);
      if (d < nearV.dist) nearV = { dist: d, name: v.name };
    });
    // Nearest functional area
    let nearF = { dist: Infinity, name: '—' };
    planState.funcMeshes.forEach((f) => {
      const d = Math.max(0, distancePointToSegment(f.pos, entry, target) - f.r);
      if (d < nearF.dist) nearF = { dist: d, name: f.name };
    });

    const dvMm = nearV.dist * 1000;
    const dfMm = nearF.dist * 1000;
    const minDist = Math.min(dvMm, dfMm);

    // Safety: 10 if minDist >= 8mm, 0 if minDist == 0
    const safety = Math.max(0, Math.min(10, minDist * 1.25));
    let verdict = 'an toàn'; let safeColor = 'var(--accent-green)';
    if (safety < 7) { verdict = 'cẩn thận'; safeColor = 'var(--accent-amber)'; }
    if (safety < 4) { verdict = 'nguy hiểm'; safeColor = 'var(--accent-red)'; }

    // Update DOM
    setText('distVessel', dvMm.toFixed(1) + ' mm');
    setText('distVesselName', nearV.name);
    setText('distFunc', dfMm.toFixed(1) + ' mm');
    setText('distFuncName', nearF.name);
    setText('probeLength', lenMm.toFixed(0) + ' mm');
    setText('safetyScore', safety.toFixed(1) + ' / 10');
    setText('safetyVerdict', verdict);
    const sScore = document.getElementById('safetyScore');
    if (sScore) sScore.style.color = safeColor;
    const sVerd = document.getElementById('safetyVerdict');
    if (sVerd) sVerd.style.color = safeColor;
  }

  function setText(id, txt) {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  }

  function setupPlanControls() {
    const az = document.getElementById('probeAzimuth');
    const el = document.getElementById('probeElevation');
    const azR = document.getElementById('probeAzimuthReadout');
    const elR = document.getElementById('probeElevationReadout');

    if (az) {
      az.addEventListener('input', (e) => {
        const deg = parseFloat(e.target.value);
        planState.azimuth = deg * Math.PI / 180;
        if (azR) azR.textContent = deg + '°';
        buildProbe();
      });
    }
    if (el) {
      el.addEventListener('input', (e) => {
        const deg = parseFloat(e.target.value);
        planState.elevation = deg * Math.PI / 180;
        if (elR) elR.textContent = (deg >= 0 ? '+' : '') + deg + '°';
        buildProbe();
      });
    }

    // Visibility toggles
    document.querySelectorAll('[data-plan-show]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-plan-show');
        const on = btn.classList.toggle('is-active');
        planState.visibility[target] = on;
        applyPlanVisibility();
      });
    });
  }

  function applyPlanVisibility() {
    if (planState.probeMesh) planState.probeMesh.visible = planState.visibility.probe;
    if (planState.probeArrow) planState.probeArrow.visible = planState.visibility.probe;
    planState.vesselMeshes.forEach((v) => v.mesh.visible = planState.visibility.vessels);
    planState.funcMeshes.forEach((f) => f.mesh.visible = planState.visibility.func);
  }

  function hidePlanPlaceholder() {
    const ph = document.getElementById('planPlaceholder');
    if (!ph) return;
    ph.style.opacity = '0';
    setTimeout(() => { ph.style.display = 'none'; }, 400);
  }

  function planOnResize() {
    if (!planState.container || !planState.renderer) return;
    const rect = planState.container.getBoundingClientRect();
    const W = rect.width || 600, H = rect.height || 460;
    planState.camera.aspect = W / H;
    planState.camera.updateProjectionMatrix();
    planState.renderer.setSize(W, H, false);
  }

  function animatePlan() {
    planState.rafId = requestAnimationFrame(animatePlan);
    if (planState.controls) planState.controls.update();

    if (planState.probeMesh) {
      const p = 0.35 + Math.sin(performance.now() / 650) * 0.18;
      planState.probeMesh.material.emissiveIntensity = p;
    }
    if (planState.tumorSphere) {
      const t = performance.now() / 800;
      const s = 1 + Math.sin(t) * 0.08;
      planState.tumorSphere.scale.set(s, s, s);
    }

    planState.renderer.render(planState.scene, planState.camera);
  }

  /* ─── Public surface ────────────────────────────────────── */
  window.AnatomyView = {
    showTumor: showTumor,
    selectLobe: selectLobe,
    setSliceMode: setSliceMode,
  };

  /* ─── Go ────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();
