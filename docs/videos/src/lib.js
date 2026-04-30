// A tiny deterministic scene toolkit. A video is one function of time:
// `start({ duration, title, draw })` installs `window.render(t)`, which redraws the whole
// SVG for second `t`. render.cjs calls it frame by frame; opening the HTML file in a
// browser just plays it.

const W = 1280;
const H = 720;
const C = {
  bg: '#0d1117',
  panel: '#161b26',
  panel2: '#1d2433',
  line: '#2c3548',
  text: '#e8ecf4',
  dim: '#8c97ad',
  teal: '#34d3c0',
  amber: '#ffb454',
  violet: '#a78bfa',
  rose: '#ff6b81',
  blue: '#5aa9ff',
  green: '#7ee787',
};
const FONT = "Inter, 'DejaVu Sans', sans-serif";
const MONO = "'DejaVu Sans Mono', monospace";

let out = [];
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, k) => a + (b - a) * k;
const E = {
  lin: (k) => k,
  out: (k) => 1 - Math.pow(1 - k, 3),
  inout: (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2),
  back: (k) => 1 + 2.70158 * Math.pow(k - 1, 3) + 1.70158 * Math.pow(k - 1, 2),
};
// eased progress of t through [a, b]
const ramp = (t, a, b, ease = E.inout) => ease(clamp((t - a) / (b - a)));
// 0 -> 1 -> 0 envelope for something visible during [a, b]
const win = (t, a, b, fin = 0.4, fout = 0.4) => clamp((t - a) / fin) * clamp((b - t) / fout);
const mix = (c1, c2, k) => {
  const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const [a, b] = [p(c1), p(c2)];
  return '#' + a.map((v, i) => Math.round(lerp(v, b[i], clamp(k))).toString(16).padStart(2, '0')).join('');
};
// deterministic pseudo-random in [0, 1)
const rnd = (n) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

const attrs = (o) =>
  `fill="${o.fill ?? 'none'}" stroke="${o.stroke ?? 'none'}" stroke-width="${o.sw ?? 1.5}" opacity="${o.op ?? 1}"` +
  (o.dash ? ` stroke-dasharray="${o.dash}"` : '') +
  (o.filter ? ` filter="url(#${o.filter})"` : '');

const rect = (x, y, w, h, o = {}) =>
  out.push(`<rect x="${x}" y="${y}" width="${Math.max(0, w)}" height="${Math.max(0, h)}" rx="${o.r ?? 0}" ${attrs(o)}/>`);
const circle = (x, y, r, o = {}) => out.push(`<circle cx="${x}" cy="${y}" r="${Math.max(0, r)}" ${attrs(o)}/>`);
const line = (x1, y1, x2, y2, o = {}) =>
  out.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-linecap="round" ${attrs({ ...o, stroke: o.stroke ?? C.line })}/>`);
const path = (d, o = {}) => out.push(`<path d="${d}" stroke-linecap="round" stroke-linejoin="round" ${attrs(o)}/>`);
const text = (x, y, s, o = {}) =>
  out.push(
    `<text x="${x}" y="${y}" font-size="${o.size ?? 20}" font-weight="${o.weight ?? 500}" fill="${o.fill ?? C.text}" ` +
      `text-anchor="${o.anchor ?? 'start'}" font-family="${o.mono ? MONO : FONT}" opacity="${o.op ?? 1}"` +
      (o.ls ? ` letter-spacing="${o.ls}"` : '') +
      `>${esc(s)}</text>`,
  );
const arrow = (x1, y1, x2, y2, o = {}) => {
  const col = o.stroke ?? C.dim;
  const a = Math.atan2(y2 - y1, x2 - x1);
  const h = o.head ?? 9;
  const bx = x2 - Math.cos(a) * h * 0.6;
  const by = y2 - Math.sin(a) * h * 0.6;
  line(x1, y1, bx, by, { ...o, stroke: col });
  const p = (d) => `${x2 - Math.cos(a + d) * h},${y2 - Math.sin(a + d) * h}`;
  out.push(`<polygon points="${x2},${y2} ${p(0.45)} ${p(-0.45)}" fill="${col}" opacity="${o.op ?? 1}"/>`);
};
const group = (tx, ty, scale, op, fn) => {
  out.push(`<g transform="translate(${tx} ${ty}) scale(${scale})" opacity="${op}">`);
  fn();
  out.push('</g>');
};
// rounded box with a centred label
const box = (x, y, w, h, label, o = {}) => {
  rect(x, y, w, h, { r: o.r ?? 10, fill: o.fill ?? C.panel2, stroke: o.stroke ?? C.line, sw: o.sw ?? 1.5, op: o.op ?? 1 });
  if (label !== '')
    text(x + w / 2, y + h / 2 + (o.size ?? 20) * 0.35, label, {
      size: o.size ?? 20, anchor: 'middle', fill: o.color ?? C.text, mono: o.mono, weight: o.weight, op: o.op ?? 1,
    });
};
const textWidth = (s, size, mono) => s.length * size * (mono ? 0.6 : 0.56);

function chrome(t, duration, title) {
  out.push(
    `<defs><radialGradient id="bg" cx="50%" cy="38%" r="80%"><stop offset="0%" stop-color="#171d2b"/>` +
      `<stop offset="100%" stop-color="${C.bg}"/></radialGradient>` +
      `<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter></defs>`,
  );
  rect(0, 0, W, H, { fill: 'url(#bg)' });
  circle(44, 36, 9, { fill: C.teal });
  circle(44, 36, 4, { fill: C.bg });
  text(62, 42, 'garlic-inference', { size: 17, weight: 700, fill: C.text });
  text(W - 36, 42, title, { size: 15, anchor: 'end', fill: C.dim });
  rect(0, H - 4, W, 4, { fill: C.panel });
  rect(0, H - 4, W * clamp(t / duration), 4, { fill: C.teal });
}

// Bottom caption, one at a time. items: [{ a, b, text }]; "\n" starts a second line.
function caption(t, items) {
  for (const it of items) {
    const op = win(t, it.a, it.b, 0.35, 0.35);
    if (op <= 0) continue;
    const lines = it.text.split('\n');
    const w = Math.max(...lines.map((l) => textWidth(l, 23))) + 56;
    const h = 20 + lines.length * 32;
    const y = H - 24 - h;
    rect(W / 2 - w / 2, y + (1 - op) * 10, w, h, { r: 14, fill: '#0b0f17', stroke: C.line, op: op * 0.94 });
    lines.forEach((l, i) =>
      text(W / 2, y + 34 + i * 32 + (1 - op) * 10, l, { size: 23, anchor: 'middle', fill: C.text, op }),
    );
  }
}

function titleCard(t, a, b, title, sub, tags) {
  const op = win(t, a, b, 0.7, 0.5);
  if (op <= 0) return;
  const dy = (1 - ramp(t, a, a + 0.9, E.out)) * 24;
  text(W / 2, 300 + dy, title, { size: 60, weight: 800, anchor: 'middle', op });
  text(W / 2, 358 + dy, sub, { size: 25, anchor: 'middle', fill: C.dim, op });
  if (tags) {
    const widths = tags.map((s) => textWidth(s, 17) + 30);
    let x = W / 2 - (widths.reduce((p, c) => p + c, 0) + 14 * (tags.length - 1)) / 2;
    tags.forEach((s, i) => {
      rect(x, 400 + dy, widths[i], 38, { r: 19, fill: C.panel2, stroke: C.line, op });
      text(x + widths[i] / 2, 425 + dy, s, { size: 17, anchor: 'middle', fill: C.teal, op });
      x += widths[i] + 14;
    });
  }
}

function start({ duration, title, draw }) {
  const svg = document.getElementById('s');
  window.DURATION = duration;
  window.render = (t) => {
    out = [];
    chrome(t, duration, title);
    draw(t);
    svg.innerHTML = out.join('');
  };
  window.render(0);
  if (!navigator.webdriver) {
    const q = new URLSearchParams(location.search);
    if (q.has('t')) return window.render(parseFloat(q.get('t')));
    const t0 = performance.now();
    const tick = () => {
      window.render(((performance.now() - t0) / 1000) % duration);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}
