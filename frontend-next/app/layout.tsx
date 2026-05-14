import type { Metadata, Viewport } from 'next';
import { Mona_Sans, JetBrains_Mono } from 'next/font/google';
import Script from 'next/script';
import './globals.css';

// Display + body — Mona Sans (GitHub open-source variable font). Distinctive
// without being saturated. Picked over Geist/Inter (training-data reflex).
const monaSans = Mona_Sans({
  subsets: ['latin', 'vietnamese'],
  display: 'swap',
  variable: '--font-sans',
  weight: ['400', '500', '600', '700', '800'],
});

// Technical / annotation labels — JetBrains Mono. The project IS technical-
// clinical (medical AI tooling) so monospace reads as voice, not costume.
const jetBrains = JetBrains_Mono({
  subsets: ['latin', 'vietnamese'],
  display: 'swap',
  variable: '--font-mono',
  weight: ['400', '500', '600'],
});

export const metadata: Metadata = {
  title: 'ADA Group — Medical AI Research',
  description:
    'Bốn pipeline AI lâm sàng đã triển khai: phát hiện động kinh từ EEG, phân đoạn u não MRI, định vị nốt phổi CT, phân tích chỉ số máu.',
  applicationName: 'ADA Group',
  authors: [{ name: 'TruongTanNghia' }],
  keywords: [
    'medical AI', 'EEG', 'seizure detection', 'BraTS', 'brain tumor',
    'lung CT', 'LIDC', 'segmentation', 'deep learning',
  ],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: 'oklch(0.10 0.005 260)',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" className={`${monaSans.variable} ${jetBrains.variable}`} suppressHydrationWarning>
      <head>
        {/* Warm up the CDN connection so the <model-viewer> ES module
            below loads as fast as possible — without this, the hero
            renders as empty boxes during the first second while DNS
            + TLS handshake to jsdelivr is in flight. */}
        <link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="anonymous" />
        <link
          rel="modulepreload"
          href="https://cdn.jsdelivr.net/npm/@google/model-viewer@4.1.0/dist/model-viewer.min.js"
          crossOrigin="anonymous"
        />
      </head>
      <body suppressHydrationWarning>
        {/* Google's <model-viewer> web component — renders GLB/GLTF
            with camera-controls, lighting, shadows, etc. beforeInteractive
            places it in <head> and registers the custom element before
            React hydrates, so <model-viewer> tags in page.tsx have a
            real implementation by the time they paint. */}
        <Script
          id="model-viewer-runtime"
          type="module"
          src="https://cdn.jsdelivr.net/npm/@google/model-viewer@4.1.0/dist/model-viewer.min.js"
          strategy="beforeInteractive"
        />
        {children}
      </body>
    </html>
  );
}
