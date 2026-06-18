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

  // Mạch máu — procedural vessels (Three.js TubeGeometry).
  // Hệ động mạch nền não + 3 cặp động mạch chính + xoang tĩnh mạch lớn.
  // Points trong toạ độ model space đã normalize. Tham khảo:
  // Circle of Willis (vòng động mạch đa giác Willis), MCA/ACA/PCA, SSS.
  let vesselsGroup = null;
  const VESSEL_PATHS = [
    {
      id: 'cow', name: 'Circle of Willis', type: 'artery', radius: 0.0022,
      points: [
        [ 0.022, -0.045,  0.002],
        [ 0.016, -0.046,  0.020],
        [ 0.000, -0.047,  0.026],
        [-0.016, -0.046,  0.020],
        [-0.022, -0.045,  0.002],
        [-0.018, -0.046, -0.016],
        [ 0.000, -0.047, -0.022],
        [ 0.018, -0.046, -0.016],
      ],
      closed: true,
    },
    {
      id: 'basilar', name: 'Basilar artery', type: 'artery', radius: 0.0020,
      points: [
        [ 0.000, -0.062, -0.018],
        [ 0.000, -0.054, -0.020],
        [ 0.000, -0.047, -0.022],
      ],
      closed: false,
    },
    {
      id: 'mca-r', name: 'MCA right', type: 'artery', radius: 0.0017,
      points: [
        [ 0.022, -0.045,  0.002],
        [ 0.038, -0.030,  0.022],
        [ 0.055, -0.012,  0.034],
        [ 0.062,  0.008,  0.030],
        [ 0.057,  0.025,  0.018],
        [ 0.045,  0.035,  0.000],
      ],
      closed: false,
    },
    {
      id: 'mca-l', name: 'MCA left', type: 'artery', radius: 0.0017,
      points: [
        [-0.022, -0.045,  0.002],
        [-0.038, -0.030,  0.022],
        [-0.055, -0.012,  0.034],
        [-0.062,  0.008,  0.030],
        [-0.057,  0.025,  0.018],
        [-0.045,  0.035,  0.000],
      ],
      closed: false,
    },
    {
      id: 'aca-r', name: 'ACA right', type: 'artery', radius: 0.0015,
      points: [
        [ 0.000, -0.045,  0.026],
        [ 0.008, -0.022,  0.046],
        [ 0.010,  0.005,  0.055],
        [ 0.012,  0.028,  0.048],
        [ 0.010,  0.045,  0.030],
        [ 0.008,  0.055,  0.008],
      ],
      closed: false,
    },
    {
      id: 'aca-l', name: 'ACA left', type: 'artery', radius: 0.0015,
      points: [
        [ 0.000, -0.045,  0.026],
        [-0.008, -0.022,  0.046],
        [-0.010,  0.005,  0.055],
        [-0.012,  0.028,  0.048],
        [-0.010,  0.045,  0.030],
        [-0.008,  0.055,  0.008],
      ],
      closed: false,
    },
    {
      id: 'pca-r', name: 'PCA right', type: 'artery', radius: 0.0015,
      points: [
        [ 0.018, -0.046, -0.016],
        [ 0.030, -0.028, -0.034],
        [ 0.034, -0.008, -0.052],
        [ 0.025,  0.012, -0.060],
        [ 0.010,  0.028, -0.055],
      ],
      closed: false,
    },
    {
      id: 'pca-l', name: 'PCA left', type: 'artery', radius: 0.0015,
      points: [
        [-0.018, -0.046, -0.016],
        [-0.030, -0.028, -0.034],
        [-0.034, -0.008, -0.052],
        [-0.025,  0.012, -0.060],
        [-0.010,  0.028, -0.055],
      ],
      closed: false,
    },
    {
      id: 'sss', name: 'Superior sagittal sinus', type: 'vein', radius: 0.0024,
      points: [
        [ 0.000,  0.040,  0.062],
        [ 0.000,  0.058,  0.040],
        [ 0.000,  0.066,  0.010],
        [ 0.000,  0.066, -0.020],
        [ 0.000,  0.055, -0.044],
        [ 0.000,  0.035, -0.062],
      ],
      closed: false,
    },
    {
      id: 'trans-r', name: 'Transverse + sigmoid right', type: 'vein', radius: 0.0018,
      points: [
        [ 0.000,  0.035, -0.062],
        [ 0.020,  0.020, -0.060],
        [ 0.035,  0.000, -0.050],
        [ 0.040, -0.020, -0.034],
        [ 0.035, -0.042, -0.020],
      ],
      closed: false,
    },
    {
      id: 'trans-l', name: 'Transverse + sigmoid left', type: 'vein', radius: 0.0018,
      points: [
        [ 0.000,  0.035, -0.062],
        [-0.020,  0.020, -0.060],
        [-0.035,  0.000, -0.050],
        [-0.040, -0.020, -0.034],
        [-0.035, -0.042, -0.020],
      ],
      closed: false,
    },
  ];

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
    buildVessels();
    setupRaycaster();
    setupAtlasCards();
    setupResetButton();
    setupSliceControls();
    setupVesselToggle();
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

    // Vessel emissive pulse — gives the arteries a "heartbeat" feel
    animateVesselsPulse();

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

  /* ─── Procedural vessels (Three.js TubeGeometry) ────────── */
  function buildVessels() {
    vesselsGroup = new THREE.Group();
    vesselsGroup.name = 'vessels';

    VESSEL_PATHS.forEach((path) => {
      const points = path.points.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const curve = new THREE.CatmullRomCurve3(
        points,
        path.closed === true,
        'catmullrom',
        0.4
      );

      const tubularSeg = path.closed ? 96 : 48;
      const radialSeg = 10;
      const tubeGeom = new THREE.TubeGeometry(
        curve, tubularSeg, path.radius, radialSeg, path.closed === true
      );

      const isArtery = path.type === 'artery';
      const mat = new THREE.MeshPhongMaterial({
        color:    isArtery ? 0xd9362a : 0x3f4794,
        emissive: isArtery ? 0x4a0e08 : 0x14163a,
        emissiveIntensity: 0.30,
        specular: isArtery ? 0xff8a7a : 0x8a9cff,
        shininess: 70,
        transparent: true,
        opacity: 0.96,
        depthWrite: false,
      });

      const mesh = new THREE.Mesh(tubeGeom, mat);
      mesh.userData.vesselId = path.id;
      mesh.userData.vesselType = path.type;
      mesh.renderOrder = 850;
      vesselsGroup.add(mesh);
    });

    scene.add(vesselsGroup);
  }

  function setupVesselToggle() {
    const btn = document.getElementById('anatomyVesselsToggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const on = btn.classList.toggle('is-active');
      if (vesselsGroup) vesselsGroup.visible = on;
    });
  }

  function animateVesselsPulse() {
    if (!vesselsGroup || !vesselsGroup.visible) return;
    const t = performance.now() / 750;
    const arteryPulse = 0.30 + Math.sin(t) * 0.12;
    const veinPulse   = 0.18 + Math.sin(t + 1.5) * 0.05;
    vesselsGroup.children.forEach((m) => {
      if (m.userData.vesselType === 'artery') {
        m.material.emissiveIntensity = arteryPulse;
      } else {
        m.material.emissiveIntensity = veinPulse;
      }
    });
  }

  /* ─── Public surface ────────────────────────────────────── */
  window.AnatomyView = {
    showTumor: showTumor,
    selectLobe: selectLobe,
    setSliceMode: setSliceMode,
    toggleVessels: function (show) {
      if (!vesselsGroup) return;
      const btn = document.getElementById('anatomyVesselsToggle');
      vesselsGroup.visible = (show !== undefined) ? !!show : !vesselsGroup.visible;
      if (btn) btn.classList.toggle('is-active', vesselsGroup.visible);
    },
  };

  /* ─── Go ────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();
