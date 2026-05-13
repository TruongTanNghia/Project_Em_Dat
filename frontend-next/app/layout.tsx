import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Medical AI Suite — EEG · MRI · CT · Blood',
  description:
    'Đa-mô-thức AI cho chẩn đoán y khoa: phát hiện động kinh từ EEG, phân đoạn u não trên MRI, đánh giá tổn thương phổi qua CT, phân tích chỉ số máu.',
  applicationName: 'Medical AI Suite',
  authors: [{ name: 'TruongTanNghia' }],
  keywords: [
    'medical AI', 'EEG', 'seizure detection', 'BraTS', 'brain tumor',
    'lung CT', 'LIDC', 'segmentation', 'deep learning', 'PyTorch', 'TensorFlow',
  ],
  openGraph: {
    title: 'Medical AI Suite',
    description: 'Multi-modal medical AI: EEG seizure detection · 3D brain tumor segmentation · Lung CT analysis',
    type: 'website',
    locale: 'vi_VN',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0a0a14',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
