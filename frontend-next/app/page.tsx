import Link from 'next/link';

type ModuleCard = {
  id: string;
  tag: string;
  title: string;
  subtitle: string;
  description: string;
  stack: string[];
  from: string;
  to: string;
  icon: React.ReactNode;
};

const MODULES: ModuleCard[] = [
  {
    id: 'eeg',
    tag: '01 · NEURO',
    title: 'EEG Seizure Detection',
    subtitle: 'CHB-MIT · CNN+BiGRU+Attention · 23 channels',
    description:
      'Phân tích điện não đồ realtime để phát hiện cơn động kinh. Mô hình hybrid CNN trích đặc trưng spatial + BiGRU bắt dependency thời gian + Attention focus vào segment quan trọng.',
    stack: ['PyTorch', 'CHB-MIT', '0.5–40 Hz', 'ROC-AUC 0.84'],
    from: '#a855f7',
    to: '#6366f1',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M2 12h3l2-6 3 12 3-8 3 4 2-2h4" />
      </svg>
    ),
  },
  {
    id: 'brain',
    tag: '02 · ONCOLOGY',
    title: 'Brain MRI Tumor Segmentation',
    subtitle: 'BraTS 2020 · 3D U-Net · 4-class NCR/ED/ET',
    description:
      'Phân đoạn u não đa mô-thức từ MRI 4 channel (FLAIR/T1/T1c/T2). Output mesh 3D mỗi class qua marching cubes, render trên brain GLB realtime với Three.js.',
    stack: ['TensorFlow', 'BraTS', '3D U-Net', '4-way TTA', 'Marching Cubes'],
    from: '#ec4899',
    to: '#8b5cf6',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M9.5 2A2.5 2.5 0 0 0 7 4.5v.07A4 4 0 0 0 4 8.5v1.13A4 4 0 0 0 2 13v.5A3.5 3.5 0 0 0 5.5 17H6v1a4 4 0 0 0 8 0v-1h.5A3.5 3.5 0 0 0 18 13.5V13a4 4 0 0 0-2-3.37V8.5a4 4 0 0 0-3-3.93V4.5A2.5 2.5 0 0 0 10.5 2h-1Z" />
        <circle cx="14" cy="10" r="1.5" fill="currentColor" />
      </svg>
    ),
  },
  {
    id: 'lung',
    tag: '03 · PULMONOLOGY',
    title: 'Lung CT Nodule Analysis',
    subtitle: 'LIDC-IDRI · DeepLabV3 · 3D + MPR views',
    description:
      'Định vị nốt phổi trên CT axial, xác định kích thước (mm), tính độ ác tính theo malignancy score, render 3D GLB phổi với vị trí tumor + so sánh axial/sagittal/coronal.',
    stack: ['PyTorch', 'DeepLabV3', 'LIDC-IDRI', 'GLB 3D'],
    from: '#06b6d4',
    to: '#3b82f6',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <path d="M12 4v14m-5 2c-3 0-5-2-5-5s1-7 4-7c2 0 3 1.5 3 3v9zm10 0c3 0 5-2 5-5s-1-7-4-7c-2 0-3 1.5-3 3v9z" />
      </svg>
    ),
  },
  {
    id: 'blood',
    tag: '04 · HEMATOLOGY',
    title: 'Blood Test Insights',
    subtitle: 'CBC + Glucose + Lipid panels · rule-based',
    description:
      'Phân tích các chỉ số xét nghiệm máu cơ bản (RBC, WBC, HGB, glucose, cholesterol...) so với reference range, đánh giá nguy cơ + đề xuất theo dõi.',
    stack: ['Rule-based', 'CBC', 'Lipid', 'Reference ranges'],
    from: '#f43f5e',
    to: '#ef4444',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
        <path d="M12 2s-7 8.5-7 13a7 7 0 0 0 14 0c0-4.5-7-13-7-13z" />
      </svg>
    ),
  },
];

export default function Home() {
  return (
    <>
      <div className="backdrop" aria-hidden />
      <div className="page">
        <div className="container">
          <nav className="nav">
            <Link href="/" className="nav-brand">
              <span className="nav-brand-mark" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                  <path d="M3 12h3l2-6 3 12 3-8 3 4 2-2h2" />
                </svg>
              </span>
              <span>Medical AI Suite</span>
            </Link>
            <div className="nav-meta">
              <span className="dot" aria-hidden />
              <span>v1.0 · 4 modules online</span>
            </div>
          </nav>

          <section className="hero">
            <div className="hero-badge">
              <span>●</span>
              <span>Multi-modal Diagnostic AI</span>
            </div>
            <h1>
              Chẩn đoán y khoa<br />
              bằng <span className="grad">AI đa-mô-thức</span>
            </h1>
            <p className="hero-sub">
              Bộ pipeline AI cho 4 chuyên khoa: điện não đồ phát hiện động kinh,
              phân đoạn u não MRI, định vị nốt phổi trên CT, và phân tích chỉ
              số máu — tất cả trên cùng một giao diện.
            </p>
            <div className="hero-meta">
              <span><strong>4</strong> modules</span>
              <span><strong>3D</strong> visualization</span>
              <span><strong>BraTS · LIDC · CHB-MIT</strong></span>
              <span><strong>PyTorch + TensorFlow</strong></span>
            </div>
          </section>

          <section className="modules">
            <div className="modules-header">
              <h2>Chọn module để bắt đầu</h2>
              <p>Click vào card → mở app phân tích</p>
            </div>

            <div className="modules-grid">
              {MODULES.map((m, i) => (
                <Link
                  key={m.id}
                  href={`/legacy.html#${m.id}`}
                  className="module-card"
                  style={{
                    // CSS variable bindings consumed by globals.css
                    ['--from' as string]: m.from,
                    ['--to' as string]: m.to,
                    ['--i' as string]: i,
                  } as React.CSSProperties}
                >
                  <div className="module-card-head">
                    <span className="module-card-icon" aria-hidden>{m.icon}</span>
                    <span className="module-card-tag">{m.tag}</span>
                  </div>
                  <h3>{m.title}</h3>
                  <div className="module-card-subtitle">{m.subtitle}</div>
                  <p className="module-card-desc">{m.description}</p>
                  <div className="module-card-stack">
                    {m.stack.map((s) => (
                      <span key={s} className="module-card-chip">{s}</span>
                    ))}
                  </div>
                  <span className="module-card-cta">
                    Mở module
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <path d="M5 12h14m-6-6 6 6-6 6" />
                    </svg>
                  </span>
                </Link>
              ))}
            </div>
          </section>

          <section className="tech-strip">
            <div className="tech-row">
              <span><strong>Frontend</strong> Next.js 16 · React 19</span>
              <span><strong>Backend</strong> Flask · PyTorch · TensorFlow</span>
              <span><strong>3D</strong> Three.js · Sketchfab · marching cubes</span>
              <span><strong>Imaging</strong> nibabel · nilearn · skimage</span>
            </div>
          </section>

          <footer className="footer">
            <span>© 2026 Medical AI Suite — by TruongTanNghia</span>
            <span>Demo only · không dùng cho chẩn đoán lâm sàng</span>
          </footer>
        </div>
      </div>
    </>
  );
}
