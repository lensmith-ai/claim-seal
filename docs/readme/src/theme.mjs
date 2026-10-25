// theme.mjs: the card and palette for the README figure (tokens, kit tone overrides, the card with its legend).
// Shared look: the anatomy kit tokens re-pointed at one dark palette: ground #0b0e13, panel #10141b,
// ink rgba(255,255,255,.92), ink2 .62, line .09, ONE accent amber #ffb020 (the kit's single "lit" tone).
export const TOKENS = {
  ground: '#0b0e13',
  panel: '#10141b',
  ink: 'rgba(255,255,255,.92)',
  ink2: 'rgba(255,255,255,.62)',
  line: 'rgba(255,255,255,.09)',
  amber: '#ffb020',
};

export const FIGURE_CSS = `
svg.iso,.iso{
  --anatomy-page:${TOKENS.ground};--anatomy-card:${TOKENS.panel};--anatomy-text:${TOKENS.ink2};--anatomy-strong:${TOKENS.ink};--anatomy-muted:${TOKENS.ink2};
  --anatomy-paper:${TOKENS.panel};--anatomy-face:#161b24;--anatomy-top:#1d2430;
  --anatomy-shade-0:#0a0d12;--anatomy-shade-1:#0d1118;--anatomy-shade-2:#11161f;--anatomy-shade-3:#161c26;
  --anatomy-lit-top:#3b2b0c;--anatomy-lit-shade:#2b1f09;
  --anatomy-hi:#e9ebef;--anatomy-mid:#7a8392;--anatomy-lo:#485161;--anatomy-faint:#2a3240;--anatomy-lit:${TOKENS.amber};
  --anatomy-halo:#020306;--anatomy-halo-opacity:.85;--anatomy-dot:#e9ebef;
  --anatomy-weight:.7px;
}
.rm-screen{fill:${TOKENS.ground};stroke:var(--anatomy-lo);stroke-width:.7px;vector-effect:non-scaling-stroke}
.rm-glass{fill:rgba(255,176,32,.13);stroke:${TOKENS.amber};stroke-width:.9px;vector-effect:non-scaling-stroke}
.rm-amber{fill:none;stroke:${TOKENS.amber};stroke-width:.8px;stroke-linecap:round;vector-effect:non-scaling-stroke}
.rm-amber-dot{fill:${TOKENS.amber}}
.iso-face-text{font-family:ui-monospace,"SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;font-weight:600;letter-spacing:.06em}
`;

// The card around a figure: panel, corners, numbered legend. Plain HTML; Georgia for the title, as the v02 page does.
export function cardHtml({
  title,
  fig,
  hint,
  readout,
  stage,
  legend,
  caption,
  kitCss = '',
  width = 1000,
}) {
  const keys = legend
    .map(
      ({ n, name, text, amber }) =>
        `<li><b class="n${amber ? ' a' : ''}">${n}</b><span><strong>${name}</strong> ${text}</span></li>`,
    )
    .join('');
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title><style>
*{box-sizing:border-box}
html,body{margin:0;background:${TOKENS.ground};color:${TOKENS.ink};-webkit-font-smoothing:antialiased}
body{font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"DejaVu Sans",sans-serif}
.card{width:${width}px;margin:0;padding:28px 28px 24px;background:${TOKENS.panel};border:1px solid ${TOKENS.line};border-radius:16px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:4px}
.fig{font:600 12px/1 ui-monospace,Menlo,Consolas,"DejaVu Sans Mono",monospace;letter-spacing:.12em;color:${TOKENS.ink2};text-transform:uppercase}
h1{margin:0;font:400 26px/1.2 Georgia,"Times New Roman",serif;color:${TOKENS.ink}}
h1 em{font-style:normal;color:${TOKENS.amber}}
.stage{position:relative;margin:6px 0 4px}
.stage svg{display:block;width:100%;height:auto}
.foot{display:flex;justify-content:space-between;font:600 11px/1 ui-monospace,Menlo,Consolas,"DejaVu Sans Mono",monospace;letter-spacing:.1em;color:${TOKENS.ink2};text-transform:uppercase;margin:2px 0 16px}
ol{list-style:none;margin:0;padding:16px 0 0;border-top:1px solid ${TOKENS.line};display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px 22px}
li{display:flex;gap:10px;align-items:flex-start;font-size:14px;line-height:1.4;color:${TOKENS.ink2}}
li strong{color:${TOKENS.ink};font-weight:600}
.n{flex:none;display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;border:1px solid rgba(255,255,255,.34);font:600 12px/1 ui-monospace,Menlo,Consolas,"DejaVu Sans Mono",monospace;color:${TOKENS.ink}}
.n.a{border-color:${TOKENS.amber};color:${TOKENS.amber}}
.cap{margin:14px 0 0;font-size:13px;color:${TOKENS.ink2}}
${kitCss}
${FIGURE_CSS}
</style></head><body><div class="card" id="card">
<div class="top"><h1>${title}</h1><span class="fig">${fig}</span></div>
<div class="stage" id="stage">${stage}</div>
<div class="foot"><span>${hint}</span><span>${readout}</span></div>
<ol>${keys}</ol>${caption ? `<p class="cap">${caption}</p>` : ''}
</div></body></html>`;
}
