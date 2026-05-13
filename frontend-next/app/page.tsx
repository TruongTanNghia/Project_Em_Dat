import { redirect } from 'next/navigation';

// The actual Medical AI app is plain HTML/CSS/JS served from
// /public/legacy.html. We use a server-side 307 redirect (NOT a
// rewrite) so the browser issues a fresh request directly to
// /legacy.html — Next.js then serves it as a pure static file,
// without injecting React hydration scripts that the legacy
// vanilla-JS code would choke on.
export default function Home() {
  redirect('/legacy.html');
}
