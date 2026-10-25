/* global console, process */
// build-how-it-works.mjs: "How claim-seal works", drawn as six stations on one bench with the anatomy figure kit.
//   ANATOMY_KIT=<anatomy kit folder> node build-how-it-works.mjs <out dir>
// Writes <out>/how-it-works.svg (standalone, each station a <g data-part>) and <out>/how-it-works-card.html
// (the card that capture-still.mjs photographs as how-it-works.png).
// Left to right, every station named from README.md and claim_seal/:
//   1 a standing sheet of writing whose lines carry citation tabs, beside a stack of source sheets;
//   2 pass 1: a card index of source titles, and a reject tray for the citation that matched nothing;
//   3 pass 2: a magnifier over the cited source, one passage outlined (the passage the judge quoted, found exactly);
//   4 each sentence: four bins, = says it, + goes further, ≠ contradicts, ? unrated;
//   5 the seal: a rack of claim-evidence rows (7, as in the offline demo) joined by one chain, an amber seal on top;
//   6 not sealed: an open tray beside the seal, inside a dashed line, holding loose cards.
// A belt carries the claims through 1 to 5. No moving spark or bead: this is a still.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { FIGURE_CSS, cardHtml } from './theme.mjs';

const KIT = process.env.ANATOMY_KIT;
if (!KIT) {
  console.error('REFUSED: set ANATOMY_KIT to the anatomy kit folder');
  process.exit(2);
}
const OUT = resolve(process.argv[2] ?? '.');
mkdirSync(OUT, { recursive: true });
const k = await import(pathToFileURL(join(KIT, 'kit/iso-kit.mjs')).href);

export const WIDTH = 1000;
export const HEIGHT = 450;

// The offline demo's numbers (README "Demo (offline)"; examples/demo-output.txt): 7 sealed rows, seal value prefix.
const DEMO_ROWS = 7;
const DEMO_SEAL = 'b49f3d11';

const BENCH = { x: -334, y: -40, w: 668, d: 84, r: 8 };
const BENCH_H = 16;
const FOOT_H = 5;
const SX = [-278, -167, -56, 55, 166, 277];
const SY = -10;
const BELT = { x: -324, y: 24, w: 548, d: 9, r: 2 };
const BELT_H = 3;
const RACK_TOP = 6 + DEMO_ROWS * 6 + 2;

const P = k.fitProjection(
  [
    ...k.boxCorners(BENCH, -BENCH_H - FOOT_H, 0),
    [SX[0] - 20, SY, 46],
    [SX[4] - 16, SY - 12, RACK_TOP + 8],
    [SX[4] + 16, SY + 12, RACK_TOP + 8],
  ],
  WIDTH,
  HEIGHT,
  { pad: 22, azimuth: 58 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const poly = (points) => k.pathOf(points.map(at), true);
const faceY = (y, x0, x1, z0, z1) =>
  poly([
    [x0, y, z0],
    [x1, y, z0],
    [x1, y, z1],
    [x0, y, z1],
  ]);
const lineY = (y, x0, x1, z) => k.pathOf([at([x0, y, z]), at([x1, y, z])]);
const byDepth = (pts) =>
  [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const part = (name, body) => `<g data-part="${name}">${body}</g>`;
const halo = (plan) =>
  `<path class="iso-halo" d="${k.haloOf(plan, 0, 0.1, P)}" style="filter:blur(3px)"/>`;
// text engraved on a +y face, centred on x, baseline at z
const faceText = (x, y, z, text, size, tone = 'hi') =>
  k.faceTextSvg(k.sideMatrix([x, y], z, P, 'left'), text, { size, tone, anchor: 'middle' });

function plateY(x0, x1, y, z0, z1, text, size) {
  const plate = { x: x0, y, w: x1 - x0, d: 0.9, r: 0.3 };
  return (
    S(k.slabOf(plate, z0, z1 - z0, P, 2), { tone: 'lo', crease: 'none' }) +
    faceText((x0 + x1) / 2, y + 0.9, z0 + (z1 - z0 - size * 0.72) / 2, text, size) +
    D([at([x0 + 1.4, y + 0.9, (z0 + z1) / 2]), at([x1 - 1.4, y + 0.9, (z0 + z1) / 2])], {
      size: 0.45,
      tone: 'mid',
    })
  );
}

// an open tray: floor, back walls, what lies inside, front walls (painter's order)
function openTray(plan, h, inside = '', tone = 'mid') {
  const t = 1.2;
  const { x, y, w, d } = plan;
  const wall = (p) => S(k.slabOf(p, 1.2, h - 1.2, P, 2, 0.2), { tone, crease: 'none' });
  return (
    S(k.slabOf(plan, 0, 1.2, P, 3, 0.3), { tone: 'lo' }) +
    wall({ x, y, w: t, d, r: 0.3 }) +
    wall({ x, y, w, d: t, r: 0.3 }) +
    inside +
    wall({ x: x + w - t, y, w: t, d, r: 0.3 }) +
    wall({ x, y: y + d - t, w, d: t, r: 0.3 })
  );
}
// a sheet lying flat, with text rules on it
function flatSheet(plan, z, rules = 4, tone = 'hi') {
  let out = S(k.slabOf(plan, z, 0.6, P, 2), { tone, crease: 'none' });
  for (let i = 0; i < rules; i++) {
    const yy = plan.y + 3 + i * ((plan.d - 5) / Math.max(1, rules - 1));
    out += L(
      k.lineOnTop([plan.x + 2.5, yy], [plan.x + plan.w - 2.5 - (i % 2) * 5, yy], z + 0.6, P),
      { tone: 'lo', free: true },
    );
  }
  return out;
}

function bench() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BENCH, -BENCH_H - FOOT_H, BENCH_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(BENCH, 20))) {
    out.push(S(k.cylinder(x, y, 8, -BENCH_H - FOOT_H, 1.6, P, 28), { tone: 'lo' }));
    out.push(
      S(k.cylinder(x, y, 6.2, -BENCH_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: 'mid' }),
    );
  }
  out.push(S(k.slabOf(BENCH, -BENCH_H, BENCH_H, P, 8, 1.6), { tone: 'mid' }));
  out.push(L(k.planOutline(k.insetPlan(BENCH, 5), 0, P), { tone: 'faint' }));
  const screws = k.corners(BENCH, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(''), { tone: 'lo' }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
  const names = [
    '1 TEXT + SOURCES',
    '2 PASS 1: FIND',
    '3 PASS 2: SUPPORT',
    '4 EACH SENTENCE',
    '5 THE SEAL',
    '6 NOT SEALED',
  ];
  SX.forEach((x, i) =>
    out.push(plateY(x - 52, x + 52, BENCH.y + BENCH.d, -14.5, -2.5, names[i], 9.0)),
  );
  // the belt that carries the claims from station 1 to the seal
  out.push(S(k.slabOf(BELT, 0, BELT_H, P, 4, 0.4), { tone: 'lo' }));
  for (let x = BELT.x + 12; x < BELT.x + BELT.w - 6; x += 12)
    out.push(
      L(k.lineOnTop([x, BELT.y + 1.5], [x, BELT.y + BELT.d - 1.5], BELT_H, P), { tone: 'faint' }),
    );
  out.push(
    D(
      [BELT.x + 4, BELT.x + BELT.w - 4].map((x) => at([x, BELT.y + BELT.d / 2, BELT_H])),
      { size: 0.7, tone: 'mid' },
    ),
  );
  return out.join('');
}

// a claim card on the belt in front of a station
const card = (x) =>
  S(k.slabOf({ x: x - 6, y: BELT.y + 1, w: 12, d: 7, r: 0.6 }, BELT_H, 1, P, 2, 0.2), {
    tone: 'hi',
    crease: 'lo',
  }) +
  L(k.lineOnTop([x - 3.5, BELT.y + 3.2], [x + 3, BELT.y + 3.2], BELT_H + 1, P), {
    tone: 'lo',
    free: true,
  }) +
  L(k.lineOnTop([x - 3.5, BELT.y + 5], [x + 1, BELT.y + 5], BELT_H + 1, P), {
    tone: 'lo',
    free: true,
  });

// 1: the writing (a standing sheet whose lines end in citation tabs) and its sources (a stack of sheets)
function input() {
  const x = SX[0];
  const out = [];
  // the sources, at the back right: three sheets, the top one ruled
  const src = { x: x + 4, y: SY - 22, w: 26, d: 20, r: 0.6 };
  out.push(halo(k.insetPlan(src, -1)));
  for (let i = 0; i < 3; i++)
    out.push(
      i < 2
        ? S(k.slabOf({ ...src, x: src.x + i * 1.2, y: src.y - i * 1.2 }, i * 1.4, 1.2, P, 2), {
            tone: 'mid',
            crease: 'none',
          })
        : flatSheet({ ...src, x: src.x + 2.4, y: src.y - 2.4 }, 2.8, 5),
    );
  // a clip on the stack
  out.push(
    S(k.slabOf({ x: src.x + 10, y: src.y - 3.2, w: 8, d: 3, r: 0.6 }, 3.4, 1.4, P, 2), {
      tone: 'lo',
    }),
  );
  // the writing: a foot and a standing sheet facing +y
  const foot = { x: x - 30, y: SY + 2, w: 30, d: 8, r: 1.4 };
  out.push(halo(foot), S(k.slabOf(foot, 0, 4, P, 3, 0.5), { tone: 'mid' }));
  out.push(L(k.planOutline({ x: x - 28, y: SY + 5, w: 26, d: 2, r: 0.6 }, 4, P, 2), { tone: 'lo' }));
  const sheet = { x: x - 28, y: SY + 5, w: 26, d: 0.8, r: 0.2 };
  out.push(S(k.slabOf(sheet, 4, 40, P, 2), { tone: 'hi', crease: 'none' }));
  const fy = sheet.y + sheet.d;
  const rows = [38, 34, 30, 26, 22, 18, 14, 10];
  rows.forEach((z, i) => {
    const end = sheet.x + sheet.w - 3 - (i % 3) * 4;
    out.push(L(lineY(fy, sheet.x + 3, end - (i % 2 ? 0 : 4), z), { tone: 'lo', free: true }));
    // a citation tab after the claim lines (every other line carries one)
    if (i % 2 === 0)
      out.push(`<path class="iso-line" data-tone="mid" d="${faceY(fy, end - 3, end, z - 1, z + 1)}"/>`);
  });
  return out.join('') + card(x);
}

// 2: pass 1, a card index of the source titles; one card raised (a title that matched); a reject tray in front
function pass1() {
  const x = SX[1];
  const box = { x: x - 26, y: SY - 18, w: 38, d: 30, r: 2 };
  const out = [halo(box), S(k.slabOf(box, 0, 14, P, 3, 0.6), { tone: 'mid' })];
  out.push(L(k.planOutline(k.insetPlan(box, 2.4), 14, P, 3), { tone: 'lo' }));
  for (let i = 0; i < 6; i++) {
    const cx = box.x + 6 + i * 5;
    const tall = i === 3;
    out.push(
      S(
        k.slabOf(
          { x: cx, y: box.y + 3.5, w: 1.1, d: box.d - 7, r: 0.3 },
          6,
          tall ? 24 : 12 + (i % 3) * 1.5,
          P,
          2,
        ),
        { tone: tall ? 'hi' : 'mid', crease: 'none' },
      ),
    );
    if (tall)
      out.push(
        S(k.slabOf({ x: cx, y: box.y + 5, w: 1.1, d: 7, r: 0.3 }, 30, 3, P, 2), {
          tone: 'hi',
          crease: 'none',
        }),
      );
  }
  out.push(L(faceY(box.y + box.d, box.x + 12, box.x + 26, 5, 10), { tone: 'lo' }));
  out.push(D([at([box.x + 19, box.y + box.d, 7.5])], { size: 0.8, tone: 'hi' }));
  // the reject tray: the citation that matched no title lies here and goes no further
  const tray = { x: x + 16, y: SY - 2, w: 20, d: 16, r: 1 };
  const reject = S(k.slabOf({ x: tray.x + 4, y: tray.y + 4, w: 12, d: 7, r: 0.6 }, 1.2, 1, P, 2, 0.2), {
    tone: 'mid',
    crease: 'lo',
  });
  out.push(halo(tray), openTray(tray, 5, reject, 'lo'));
  return out.join('') + card(x);
}

// 3: pass 2, a magnifier over the cited source; the passage the judge quoted is outlined (found exactly)
function pass2() {
  const x = SX[2];
  const sheet = { x: x - 10, y: SY - 17, w: 32, d: 28, r: 0.5 };
  const out = [flatSheet(sheet, 0, 6)];
  // the located passage
  out.push(
    L(
      k.planOutline({ x: sheet.x + 2, y: sheet.y + 10.4, w: 22, d: 4.4, r: 0.6 }, 0.6, P, 2),
      { tone: 'hi' },
    ),
  );
  const post = [x - 22, SY + 8];
  out.push(halo({ x: post[0] - 9, y: post[1] - 9, w: 18, d: 18, r: 9 }));
  out.push(S(k.cylinder(post[0], post[1], 9, 0, 4, P, 32, 0.4), { tone: 'mid' }));
  out.push(S(k.cylinder(post[0], post[1], 2, 4, 36, P, 18), { tone: 'mid', crease: 'none' }));
  out.push(S(k.cylinder(post[0], post[1], 3.2, 36, 5, P, 20), { tone: 'mid' }));
  out.push(
    S(k.slabOf({ x: post[0], y: post[1] - 1.6, w: 20, d: 3.2, r: 1.2 }, 37, 2.6, P, 2), {
      tone: 'mid',
      crease: 'none',
    }),
  );
  const lens = [x + 6, SY - 3];
  out.push(S(k.cylinder(lens[0], lens[1], 13, 34, 3, P, 56, 0.5), { tone: 'mid' }));
  out.push(
    `<path d="${k.ring(lens[0], lens[1], 10.5, 37, P, 56)}" style="fill:rgba(255,255,255,.07);stroke:var(--anatomy-hi);stroke-width:.7px;vector-effect:non-scaling-stroke"/>`,
  );
  out.push(L(k.ring(lens[0], lens[1], 11.6, 37, P, 56), { tone: 'lo' }));
  out.push(
    L(k.lineOnTop([lens[0] - 6, lens[1] - 4], [lens[0] - 2, lens[1] - 7.5], 37, P), {
      tone: 'hi',
      free: true,
    }),
  );
  return out.join('') + card(x);
}

// 4: each sentence sorted into one of four bins; glyphs on the bins, words in the legend
function sentences() {
  const x = SX[3];
  const glyphs = ['=', '+', '≠', '?'];
  const out = [halo({ x: x - 49, y: SY - 15, w: 98, d: 29, r: 2 })];
  out.push(S(k.slabOf({ x: x - 50, y: SY - 16, w: 100, d: 31, r: 2 }, 0, 2, P, 3, 0.4), { tone: 'mid' }));
  glyphs.forEach((g, i) => {
    const bx = x - 48 + i * 24.5;
    const bin = { x: bx, y: SY - 14, w: 23, d: 27, r: 0.8 };
    const inside = S(
      k.slabOf({ x: bx + 5, y: SY - 7, w: 13, d: 9, r: 0.6 }, 1.2, 1, P, 2, 0.2),
      { tone: 'hi', crease: 'lo' },
    );
    out.push(
      `<g transform="${k.translateAlong(P, [0, 0, 2])}">${openTray(bin, 12, inside)}${faceText(bx + 11.5, bin.y + bin.d, 1.8, g, 10.5)}</g>`,
    );
  });
  return out.join('') + card(x);
}

// 5: the seal, a rack of the claim-evidence rows joined by one chain, the seal value on top
function seal() {
  const x = SX[4];
  const base = { x: x - 18, y: SY - 14, w: 36, d: 28, r: 2 };
  const out = [halo(base), S(k.slabOf(base, 0, 6, P, 3, 0.6), { tone: 'mid' })];
  const postAt = [
    [base.x + 2, base.y + 2],
    [base.x + base.w - 5, base.y + 2],
    [base.x + 2, base.y + base.d - 5],
    [base.x + base.w - 5, base.y + base.d - 5],
  ];
  const post = ([px, py]) =>
    S(k.slabOf({ x: px, y: py, w: 3, d: 3, r: 0.6 }, 6, RACK_TOP - 6, P, 2), {
      tone: 'mid',
      crease: 'none',
    });
  const back = byDepth(postAt).slice(0, 2);
  const front = byDepth(postAt).slice(2);
  back.forEach((p) => out.push(post(p)));
  const row = { x: base.x + 3.5, y: base.y + 3.5, w: base.w - 7, d: base.d - 7, r: 0.8 };
  for (let i = 0; i < DEMO_ROWS; i++) {
    const z = 8 + i * 6;
    out.push(S(k.slabOf(row, z, 2.2, P, 2, 0.3), { tone: i === DEMO_ROWS - 1 ? 'hi' : 'mid' }));
    out.push(
      L(k.lineOnTop([row.x + 3, row.y + 4], [row.x + row.w - 9, row.y + 4], z + 2.2, P), {
        tone: 'lo',
        free: true,
      }),
    );
  }
  front.forEach((p) => out.push(post(p)));
  // the cap and the chain value
  out.push(
    S(k.slabOf(k.insetPlan(base, 1), RACK_TOP, 2.4, P, 3, 0.5), { tone: 'mid' }),
  );
  const c = [x, SY];
  out.push(S(k.cylinder(c[0], c[1], 9, RACK_TOP + 2.4, 2.6, P, 40, 0.4), { tone: 'hi', lit: true }));
  out.push(`<path class="rm-amber" d="${k.ring(c[0], c[1], 6.4, RACK_TOP + 5, P, 40)}"/>`);
  out.push(
    `<circle class="rm-amber-dot" cx="${at([c[0], c[1], RACK_TOP + 5])[0].toFixed(1)}" cy="${at([c[0], c[1], RACK_TOP + 5])[1].toFixed(1)}" r="1"/>`,
  );
  // the chain down the front face: one link between each pair of rows, row k's hash feeds row k+1
  const fy = row.y + row.d + 0.1;
  const cx = row.x + row.w - 5;
  for (let i = 0; i < DEMO_ROWS; i++) {
    const z = 9.1 + i * 6;
    out.push(`<path class="rm-amber" d="${k.sideRing([cx, fy, z], 1.6, 'y', P, 20)}"/>`);
    if (i < DEMO_ROWS - 1)
      out.push(
        `<path class="rm-amber" d="${k.pathOf([at([cx, fy, z + 1.6]), at([cx, fy, z + 4.4])])}"/>`,
      );
  }
  out.push(
    `<path class="rm-amber" d="${k.pathOf([at([cx, fy, 9.1 + (DEMO_ROWS - 1) * 6 + 1.6]), at([cx, fy, RACK_TOP])])}"/>`,
  );
  out.push(faceText(x, base.y + base.d, 1.4, 'SHA-256', 3.4, 'mid'));
  return out.join('') + card(x);
}

// 6: not sealed, an open tray beside the seal, inside a dashed line, loose cards in it
function notSealed() {
  const x = SX[5];
  const out = [];
  out.push(
    L(k.planOutline({ x: x - 40, y: SY - 24, w: 76, d: 48, r: 3 }, 0, P, 3), {
      tone: 'mid',
      dashed: true,
    }),
  );
  const tray = { x: x - 30, y: SY - 16, w: 56, d: 32, r: 1.2 };
  const loose =
    flatSheet({ x: x - 26, y: SY - 12, w: 18, d: 12, r: 0.4 }, 1.2, 3, 'mid') +
    flatSheet({ x: x - 6, y: SY - 13, w: 14, d: 10, r: 0.4 }, 1.2, 3, 'mid') +
    flatSheet({ x: x + 10, y: SY - 11, w: 12, d: 9, r: 0.4 }, 1.2, 2, 'mid') +
    flatSheet({ x: x - 22, y: SY + 2, w: 16, d: 10, r: 0.4 }, 1.2, 3, 'mid') +
    flatSheet({ x: x - 2, y: SY + 1, w: 22, d: 11, r: 0.4 }, 1.2, 3, 'mid') +
    flatSheet({ x: x - 4, y: SY - 4, w: 13, d: 9, r: 0.4 }, 1.8, 2, 'hi');
  out.push(halo(tray), openTray(tray, 6, loose, 'mid'));
  return out.join('');
}

const PARTS = [
  ['bench', bench()],
  ['input', input()],
  ['pass1', pass1()],
  ['pass2', pass2()],
  ['sentences', sentences()],
  ['seal', seal()],
  ['not-sealed', notSealed()],
];

const LABEL =
  'How claim-seal works, drawn as six stations on one bench. 1, a standing sheet of writing whose lines end in citation tabs, beside a stack of source sheets. 2, pass 1: a card index of source titles with one card raised, and a reject tray holding a citation that matched nothing. 3, pass 2: a magnifier over the cited source, one passage outlined. 4, four bins, one mark per sentence and source: = says it, + goes further, ≠ contradicts, ? unrated. 5, the seal: seven rows on a rack joined by an amber chain, an amber seal on top. 6, an open tray of loose cards inside a dashed line: what the seal does not cover.';
const body = PARTS.map(([name, svg]) => part(name, svg)).join('');
const svgStandalone = k
  .figureSvg({ width: WIDTH, height: HEIGHT, label: LABEL, body, standalone: 'dark' })
  .replace('</style>', `${FIGURE_CSS}</style>`);
writeFileSync(join(OUT, 'how-it-works.svg'), svgStandalone);

export const LEGEND = [
  {
    n: 1,
    name: 'Text + sources.',
    text: 'Writing whose claims cite sources by title, plus the sources, each fingerprinted with SHA-256.',
  },
  {
    n: 2,
    name: 'Pass 1: find.',
    text: 'Plain code, no model. Each cited title must match a source you gave. No match: “unresolved”, and it stops.',
  },
  {
    n: 3,
    name: 'Pass 2: support.',
    text: 'One model call per claim: supported, partial or unsupported. Code keeps a quote only if it is found exactly.',
  },
  {
    n: 4,
    name: 'Each sentence.',
    text: 'Marked against each source: = says what it says, + goes further, ≠ contradicts, ? unrated.',
  },
  {
    n: 5,
    name: 'The seal.',
    text: 'One row per claim per citation: outcome, where the passage sits (never its text), verb. Each row’s hash feeds the next: edit a row and the check fails.',
    amber: true,
  },
  {
    n: 6,
    name: 'Not sealed.',
    text: 'Claim verdicts and their reasons, the sentence labels from 4, overclaim flags, rule ids. Changing them still verifies.',
  },
];
const stage = k
  .figureSvg({ width: WIDTH, height: HEIGHT, label: LABEL, body, className: 'iso' })
  .replace('<svg', '<svg data-theme="dark"');
writeFileSync(
  join(OUT, 'how-it-works-card.html'),
  cardHtml({
    title: 'How claim-seal <em>works</em>',
    fig: 'Fig 1',
    hint: 'Pass 1 is plain code · pass 2 asks a model',
    readout: `Demo: ${DEMO_ROWS} rows · seal ${DEMO_SEAL}`,
    stage: `<div class="iso" data-theme="dark">${stage}</div>`,
    legend: LEGEND,
    caption:
      'The chain has no key: whoever rewrites the rows can redo it, so it counts only next to version history. It never says a claim is true, only whether the given sources back it.',
    kitCss: k.ISO_CSS + 'li{font-size:15.5px}.cap{font-size:14px}',
  }),
);
console.log(
  `wrote ${OUT}/how-it-works.svg (${svgStandalone.length} B), how-it-works-card.html; parts: ${PARTS.map(([n]) => n).join(', ')}`,
);
