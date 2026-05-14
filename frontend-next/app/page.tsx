import Link from 'next/link';

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
        interaction-prompt="none"
        exposure="0.9"
        shadow-intensity="0.6"
        shadow-softness="1"
        tone-mapping="neutral"
        camera-orbit="-25deg 78deg 1.6m"
        min-camera-orbit="auto 60deg auto"
        max-camera-orbit="auto 100deg auto"
        loading="eager"
        reveal="auto"
        touch-action="pan-y"
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: 'transparent',
          '--poster-color': 'transparent',
        } as React.CSSProperties}
      />

      {/* HUD corner stamps */}
      <span className="scan-corner tl"><b>BraTS-2021</b> · case 00621</span>
      <span className="scan-corner tr">FLAIR · ax<br /><b>z = 76</b></span>
      <span className="scan-corner bl">3D U-Net · TF<br /><b>Dice WT 0.83</b></span>
      <span className="scan-corner br"><b>R</b> · L</span>

      {/* Annotation pills — positioned over the rotating model */}
      <span className="annot" style={{ top: '22%', right: '4%' }}>
        <span className="annot-tick" />
        NCR · <b>12.3 cm³</b>
      </span>
      <span className="annot" style={{ top: '46%', right: '4%' }}>
        <span className="annot-tick" />
        ET ring · <b>4.2 cm³</b>
      </span>
      <span className="annot" style={{ bottom: '18%', left: '6%' }}>
        <span className="annot-tick" />
        ED edema · <b>11.9 cm³</b>
      </span>
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
        interaction-prompt="none"
        exposure="0.85"
        shadow-intensity="0.5"
        tone-mapping="neutral"
        camera-orbit="35deg 75deg 1.4m"
        loading="lazy"
        reveal="auto"
        touch-action="pan-y"
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' } as React.CSSProperties}
      />
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
        rotation-per-second="14deg"
        camera-controls
        disable-zoom
        interaction-prompt="none"
        exposure="1.0"
        shadow-intensity="0.4"
        tone-mapping="neutral"
        camera-orbit="-20deg 80deg 2.4m"
        loading="lazy"
        reveal="auto"
        touch-action="pan-y"
        style={{ width: '100%', height: '100%', backgroundColor: 'transparent' } as React.CSSProperties}
      />
      <span className="scan-corner tl"><b>LIDC-IDRI</b> · ax slice 84</span>
      <span className="scan-corner tr">DeepLabV3<br /><b>ø 18 mm</b></span>
      <span className="scan-corner bl">malignancy 4/5</span>
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

/* ─── Page ─────────────────────────────────────────────────────── */

export default function Home() {
  return (
    <div className="page">
      <header className="frame">
        <div className="registry">
          <Link href="/" className="brand">
            ADA Group
            <span className="brand-sub">Medical AI · Research</span>
          </Link>
          <div /> {/* spacer */}
          <div className="registry-meta">
            <span className="pulse" />
            <span>4 pipelines online · v1.0</span>
          </div>
        </div>
      </header>

      <main>
        <section className="frame hero">
          <HeroScan />

          <div>
            <p className="hero-eyebrow reveal">
              Bốn pipeline lâm sàng đã triển khai
            </p>
            <h1 className="hero-title reveal" style={{ ['--delay' as string]: '0.05s' }}>
              Đọc lát cắt.<br />
              <em>Định vị</em> tổn thương.<br />
              Trong vài giây.
            </h1>
            <p className="hero-sub reveal" style={{ ['--delay' as string]: '0.15s' }}>
              ADA Group dựng pipeline AI cho 4 chuyên khoa,
              chạy trên dataset chuẩn (BraTS, LIDC, CHB-MIT) và phơi
              kết quả qua một giao diện duy nhất. Mỗi case bên dưới
              là pipeline thật, không phải mockup.
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
              <dd>Next.js · PyTorch · TF</dd>
              <dt>Datasets</dt>
              <dd>BraTS · LIDC · CHB-MIT</dd>
              <dt>Modules</dt>
              <dd>EEG · MRI · CT · Blood</dd>
              <dt>3D viz</dt>
              <dd>Three.js · marching cubes</dd>
            </dl>
          </div>
        </section>

        <section className="frame cases">
          <div className="cases-header">
            <h2>Bốn case, bốn pipeline, một phòng đọc duy nhất.</h2>
            <p>
              Mỗi case dưới đây là pipeline production đã trained.
              Click để mở demo trực tiếp với dataset thật.
            </p>
          </div>

          <article className="case" data-case="01">
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

          <article className="case" data-case="02">
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

          <article className="case" data-case="03">
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

          <article className="case" data-case="04">
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
        </section>
      </main>

      <footer className="frame footer">
        <span>© 2026 ADA Group · TruongTanNghia</span>
        <span className="footer-warn">Demo · không dùng cho chẩn đoán lâm sàng</span>
      </footer>
    </div>
  );
}
