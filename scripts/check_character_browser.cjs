// SPDX-License-Identifier: MIT
// Offline browser regression for a generated fixture: 14 tracks, shots at frames 0 and 90.
// Run through scripts.run_browser_checks so the project-owned Chromium is used.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({
    headless: true,
  });
  try {
    const page = await browser.newPage();
    const origin = new URL(process.argv[2]).origin;
    const externalRequests = [];
    await page.route("**/*", (route) => {
      const url = new URL(route.request().url());
      if (url.origin === origin) return route.continue();
      externalRequests.push(url.href);
      return route.abort();
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.argv[2]);
    await page.waitForSelector("canvas");
    await page.evaluate(() => fetch("https://example.invalid/offline-probe").catch(() => null));
    assert.deepEqual(
      externalRequests.splice(0),
      ["https://example.invalid/offline-probe"],
      "off-origin requests are intercepted before leaving the browser",
    );
    await page.waitForFunction(() => document.querySelector("#video").readyState >= 1);
    await page.click("#play");
    await page.locator("#scrub").evaluate((el) => {
      el.value = "35";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    try {
      await page.waitForFunction(
        () => {
          const video = document.querySelector("#video");
          return video.currentTime > 0 && !video.seeking;
        },
        null,
        { timeout: 5000 },
      );
    } catch (error) {
      console.error("video seek state", await page.locator("#video").evaluate((video) => ({
        currentTime: video.currentTime,
        duration: video.duration,
        readyState: video.readyState,
        networkState: video.networkState,
        seeking: video.seeking,
        error: video.error?.code,
        buffered: Array.from({ length: video.buffered.length }, (_, i) => [video.buffered.start(i), video.buffered.end(i)]),
        seekable: Array.from({ length: video.seekable.length }, (_, i) => [video.seekable.start(i), video.seekable.end(i)]),
      })));
      throw error;
    }
    const observed = await page.evaluate(() => ({
      cast: document.querySelectorAll("#cast select").length,
      videoTime: document.querySelector("#video").currentTime,
      videoReadyState: document.querySelector("#video").readyState,
      videoDuration: document.querySelector("#video").duration,
      videoError: document.querySelector("#video").error?.code,
      videoSource: document.querySelector("#video").currentSrc,
      frame: window.__stage.current.frame,
      targetTime: (() => { const s = window.__stage.stage.meta.shots[1]; return (s.frame0 + window.__stage.current.frame - s.global0) / window.__stage.stage.meta.fps; })(),
      shot: document.querySelector("#shot").textContent,
    }));
    console.log(JSON.stringify({ observed, errors }));
    const failures = [];
    try {
      assert.equal(observed.cast, 14, "every track has a recast control");
    } catch (e) {
      failures.push(e.message);
    }
    try {
      assert(
        Math.abs(observed.videoTime - 95 / 30) < 0.13,
        "source video uses original shot offset",
      );
    } catch (e) {
      failures.push(e.message);
    }
    await page.locator("#scrub").evaluate((el) => {
      el.value = "25";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.selectOption("#speed", "0.5");
    await page.click("#play");
    // Pause in the same browser task that observes the crossing: this short fixture
    // can wrap while a separate Playwright click waits for an animation frame.
    await page.waitForFunction(
      () => {
        const viewer = window.__stage;
        if (viewer.current.frame < 35) return false;
        viewer.current.playing = false;
        viewer.render(true);
        return true;
      },
      null,
      { polling: "raf" },
    );
    const crossing = await page.evaluate(() => ({
      frame: Number(document.querySelector("#scrub").value),
      videoTime: document.querySelector("#video").currentTime,
      rate: document.querySelector("#video").playbackRate,
    }));
    console.log(JSON.stringify({ crossing }));
    assert(
      crossing.frame >= 35 && crossing.frame < 60,
      "snapshot is in the second shot before wrapping",
    );
    try {
      assert(
        Math.abs(crossing.videoTime - (90 + crossing.frame - 30) / 30) < 0.2,
        "continuous playback follows original time across cut",
      );
    } catch (e) {
      failures.push(e.message);
    }
    assert.equal(
      crossing.rate,
      0.5,
      "source video follows selected playback speed",
    );
    assert.equal(errors.length, 0, "no browser runtime errors");
    assert.deepEqual(externalRequests, [], "browser made no external requests");
    assert.deepEqual(failures, [], "browser regression failures");
    console.log(
      "PASS: all tracks recastable; scrub and continuous playback synchronized across omitted interval",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
