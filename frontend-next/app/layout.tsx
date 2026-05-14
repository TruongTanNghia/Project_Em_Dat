import type { Metadata, Viewport } from 'next';
import { Mona_Sans, JetBrains_Mono } from 'next/font/google';
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
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
