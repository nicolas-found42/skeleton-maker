// SPDX-License-Identifier: MIT
// Offline browser regression for a generated fixture: 14 tracks, shots at frames 0 and 90.
// Run: PLAYWRIGHT_MODULE=/path/to/playwright node scripts/check_character_browser.cjs URL
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.argv[2]);
    await page.waitForSelector('canvas');
    await page.click('#play');
    await page.locator('#scrub').evaluate(el => {
      el.value = '35'; el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForFunction(() => !document.querySelector('#video').seeking);
    const observed = await page.evaluate(() => ({
      cast: document.querySelectorAll('#cast select').length,
      videoTime: document.querySelector('#video').currentTime,
      shot: document.querySelector('#shot').textContent,
    }));
    console.log(JSON.stringify({ observed, errors }));
    const failures = [];
    try { assert.equal(observed.cast, 14, 'every track has a recast control'); } catch (e) { failures.push(e.message); }
    try { assert(Math.abs(observed.videoTime - 95 / 30) < .13, 'source video uses original shot offset'); } catch (e) { failures.push(e.message); }
    await page.locator('#scrub').evaluate(el => {
      el.value = '25'; el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.selectOption('#speed', '0.5');
    await page.click('#play');
    // Pause in the same browser task that observes the crossing: this short fixture
    // can wrap while a separate Playwright click waits for an animation frame.
    await page.waitForFunction(() => {
      const viewer = window.__stage;
      if (viewer.current.frame < 35) return false;
      viewer.current.playing = false;
      viewer.render(true);
      return true;
    }, null, { polling: 'raf' });
    await page.waitForFunction(() => !document.querySelector('#video').seeking);
    const crossing = await page.evaluate(() => ({
      frame: Number(document.querySelector('#scrub').value),
      videoTime: document.querySelector('#video').currentTime,
      rate: document.querySelector('#video').playbackRate,
    }));
    console.log(JSON.stringify({ crossing }));
    assert(crossing.frame >= 35 && crossing.frame < 60, 'snapshot is in the second shot before wrapping');
    try { assert(Math.abs(crossing.videoTime - (90 + crossing.frame - 30) / 30) < .2, 'continuous playback follows original time across cut'); } catch (e) { failures.push(e.message); }
    assert.equal(crossing.rate, .5, 'source video follows selected playback speed');
    assert.equal(errors.length, 0, 'no browser runtime errors');
    assert.deepEqual(failures, [], 'browser regression failures');
    console.log('PASS: all tracks recastable; scrub and continuous playback synchronized across omitted interval');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
