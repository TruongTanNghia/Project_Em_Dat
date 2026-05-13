import type { Metadata, Viewport } from 'next';

export const metadata: Metadata = {
  title: 'Medical AI Suite',
  description: 'EEG · Brain MRI · Lung CT · Blood — multi-module AI diagnostics',
  applicationName: 'Medical AI Suite',
  authors: [{ name: 'TruongTanNghia' }],
  icons: {
    icon: '/favicon.ico',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: '#0a0a14',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
