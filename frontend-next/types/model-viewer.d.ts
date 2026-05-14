// Tell TypeScript about the <model-viewer> custom element from Google.
// We load the runtime script via next/script in app/layout.tsx; this
// file only widens JSX typings so TSX files can use the element.

import 'react';

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement> & {
          src?: string;
          alt?: string;
          poster?: string;
          'auto-rotate'?: boolean;
          'auto-rotate-delay'?: number | string;
          'rotation-per-second'?: string;
          'camera-controls'?: boolean;
          'disable-zoom'?: boolean;
          'disable-pan'?: boolean;
          'interaction-prompt'?: 'auto' | 'when-focused' | 'none';
          'shadow-intensity'?: number | string;
          'shadow-softness'?: number | string;
          exposure?: number | string;
          'environment-image'?: string;
          'skybox-image'?: string;
          'tone-mapping'?: 'neutral' | 'aces' | 'commerce' | 'none';
          loading?: 'auto' | 'lazy' | 'eager';
          reveal?: 'auto' | 'interaction' | 'manual';
          'camera-orbit'?: string;
          'min-camera-orbit'?: string;
          'max-camera-orbit'?: string;
          'field-of-view'?: string;
          'min-field-of-view'?: string;
          'max-field-of-view'?: string;
          'touch-action'?: string;
        },
        HTMLElement
      >;
    }
  }
}
