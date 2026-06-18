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

    // Lighting — soft key + cool fill + warm rim
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));

    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(1, 1.2, 0.8);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0x88a8ff, 0.32);
    fill.position.set(-1, -0.4, 0.5);
    scene.add(fill);

    const rim = new THREE.DirectionalLight(0xa78bfa, 0.28);
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

        // Center model so origin is roughly centroid
        const box = new THREE.Box3().setFromObject(brainModel);
        const center = box.getCenter(new THREE.Vector3());
        brainModel.position.sub(center);

        // Make tissue semi-transparent so hotspots inside read through
        brainModel.traverse((node) => {
          if (node.isMesh && node.material) {
            node.material = node.material.clone();
            node.material.transparent = true;
            node.material.opacity = 0.70;
            node.material.depthWrite = false;
            node.material.side = THREE.DoubleSide;
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

  /* ─── Public surface ────────────────────────────────────── */
  window.AnatomyView = {
    showTumor: showTumor,
    selectLobe: selectLobe,
  };

  /* ─── Go ────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap);
  } else {
    bootstrap();
  }
})();
