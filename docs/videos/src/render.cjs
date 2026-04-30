// Render a scene to MP4 (and a poster frame):
//   NODE_PATH=$(npm root -g) node render.cjs <scene> [--fps 30] [--poster 12.5] [--still 7.5]
// Needs `playwright` with a Chromium (set CHROMIUM=/path/to/chrome to use an existing one) and ffmpeg.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const args = process.argv.slice(2);
const scene = args[0];
const opt = (name, dflt) => (args.includes(name) ? parseFloat(args[args.indexOf(name) + 1]) : dflt);
const fps = opt('--fps', 30);
const poster = opt('--poster', null);
const still = opt('--still', null);

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM || undefined,
    args: ['--no-sandbox', '--font-render-hinting=none'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto('file://' + path.join(__dirname, `${scene}.html`));
  await page.waitForFunction(() => typeof window.render === 'function');
  await page.evaluate(() => document.fonts.ready);
  const duration = await page.evaluate(() => window.DURATION);
  const videos = path.join(__dirname, '..');

  const shot = async (t, file) => {
    await page.evaluate((t) => window.render(t), t);
    fs.writeFileSync(file, await page.screenshot({ type: 'png' }));
  };
  if (still !== null) {
    await shot(still, `/tmp/${scene}-${still}.png`);
    console.log(`/tmp/${scene}-${still}.png`);
    return browser.close();
  }

  const mp4 = path.join(videos, `${scene}.mp4`);
  const ff = spawn('ffmpeg', [
    '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4,
  ]);
  ff.stderr.pipe(process.stderr);
  const frames = Math.round(duration * fps);
  for (let i = 0; i < frames; i++) {
    await page.evaluate((t) => window.render(t), i / fps);
    const png = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 150 === 0) process.stdout.write(`\r${scene}: ${i}/${frames}`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  if (poster !== null) await shot(poster, path.join(videos, 'posters', `${scene}.png`));
  console.log(`\r${scene}: ${mp4} (${(fs.statSync(mp4).size / 1e6).toFixed(1)} MB)`);
  await browser.close();
})();
