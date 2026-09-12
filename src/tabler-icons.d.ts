declare module '@tabler/icons-react/dist/esm/icons/*.mjs' {
  import type { ForwardRefExoticComponent, RefAttributes, SVGProps } from 'react';
  const Icon: ForwardRefExoticComponent<Omit<SVGProps<SVGSVGElement>, 'stroke'> & { size?: number | string; stroke?: number | string } & RefAttributes<SVGSVGElement>>;
  export default Icon;
}
