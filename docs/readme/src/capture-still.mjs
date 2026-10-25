/* global console, process, document */
// capture-still.mjs: photograph the card page as a PNG at 2x (a 1000 px card becomes a 2000 px picture).
//   PLAYWRIGHT_MODULES=<node_modules with playwright> node capture-still.mjs <card.html> <out.png>
// The page is opened from disk; any request that leaves the file is counted and fails the capture.
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [card, out] = process.argv.slice(2).map((p) => resolve(p));
if (!card || !out) {
  console.error('usage: capture-still.mjs <card.html> <out.png>');
  process.exit(2);
}
const require = createRequire(join(resolve(process.env.PLAYWRIGHT_MODULES ?? '.'), 'noop.js'));
const { chromium } = require('playwright');
const browser = await chromium.launch();
const outside = [];
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 });
  page.on('request', (r) => (r.url().startsWith('file:') ? 0 : outside.push(r.url())));
  await page.goto(pathToFileURL(card).href);
  await page.evaluate(() => document.fonts.ready);
  await page.locator('#card').screenshot({ path: out });
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
}
console.log(`outside requests: ${outside.length}${outside.length ? ' ' + outside.join(' ') : ''}`);
if (outside.length) process.exit(1);
