// SPDX-License-Identifier: MIT
// Public offline viewer regression with known mask pixels, source frame timing and a shot cut.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const origin = new URL(process.argv[2]).origin;
    const externalRequests = [];
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== origin) externalRequests.push(request.url());
    });
    await page.route("**/*", (route) => {
      const url = new URL(route.request().url());
      return url.origin === origin ? route.continue() : route.abort();
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.argv[2]);
    await page.waitForFunction(() => document.querySelector("#video").readyState >= 1);
    await page.waitForFunction(() => {
      const ctx = document.querySelector("#overlay").getContext("2d");
      return ctx.getImageData(30, 40, 1, 1).data[3] > 0;
    });

    const repaintReads = await page.evaluate(() => {
      const original = CanvasRenderingContext2D.prototype.getImageData;
      let reads = 0;
      CanvasRenderingContext2D.prototype.getImageData = function (...args) {
        if (args[2] > 1 && args[3] > 1) reads += 1;
        return original.apply(this, args);
      };
      try {
        const input = document.querySelector("#entity");
        for (let i = 0; i < 20; i += 1) input.dispatchEvent(new Event("change"));
        return reads;
      } finally {
        CanvasRenderingContext2D.prototype.getImageData = original;
      }
    });
    assert.equal(repaintReads, 0, "repainting loaded masks must reuse tinted pixels");

    const selected = "shot-0/chair";
    await page.selectOption("#entity", selected);
    const saved = await page.evaluate(() => {
      const pixel = (x, y) => Array.from(document.querySelector("#overlay").getContext("2d").getImageData(x, y, 1, 1).data);
      return {
        maskPixel: pixel(30, 40),
        maskOutsidePixel: pixel(150, 40),
        skeletonPixel: pixel(310, 90),
        entityText: document.querySelector("#detail").innerText,
        videoDuration: document.querySelector("#video").duration,
        entityOptionCount: document.querySelectorAll("#entity option").length,
      };
    });
    assert(saved.maskPixel[1] > saved.maskPixel[0], `known floor pixel should be green: ${saved.maskPixel}`);
    assert.equal(saved.maskOutsidePixel[3], 0, `outside mask pixel should remain transparent: ${saved.maskOutsidePixel}`);
    assert(saved.skeletonPixel[0] > 0 && saved.skeletonPixel[3] > 0, `known skeleton pixel should render: ${saved.skeletonPixel}`);
    assert.match(saved.entityText, /chair/);
    assert.match(saved.entityText, /Visibility\nvisible/);
    assert.match(saved.entityText, /Confidence\n0\.92/);
    assert.match(saved.entityText, /Source frames\n0, 15/);
    assert.equal(saved.videoDuration, 5);
    assert.equal(saved.entityOptionCount, 5, "surface, object and vehicle entities are selectable");
    fs.mkdirSync(path.dirname(process.argv[3]), { recursive: true });
    await page.screenshot({ path: process.argv[3], fullPage: true });

    await page.locator('[aria-label="Show surfaces"]').uncheck();
    const hidden = await page.evaluate(() => document.querySelector("#overlay").getContext("2d").getImageData(30, 40, 1, 1).data[3]);
    assert.equal(hidden, 0, "surface family toggle hides its mask");
    await page.locator('[aria-label="Show surfaces"]').check();
    await page.locator('[aria-label="Show objects"]').uncheck();
    assert.equal(
      await page.evaluate(() => document.querySelector("#overlay").getContext("2d").getImageData(140, 120, 1, 1).data[3]),
      0,
      "object family toggle hides its box",
    );
    await page.locator('[aria-label="Show objects"]').check();
    await page.locator('[aria-label="Show vehicles"]').uncheck();
    assert.equal(
      await page.evaluate(() => document.querySelector("#overlay").getContext("2d").getImageData(400, 120, 1, 1).data[3]),
      0,
      "vehicle family toggle hides its box",
    );
    await page.locator('[aria-label="Show vehicles"]').check();
    await page.locator('[aria-label="Show people"]').uncheck();
    assert.equal(
      await page.evaluate(() => document.querySelector("#overlay").getContext("2d").getImageData(310, 90, 1, 1).data[3]),
      0,
      "people family toggle hides its skeleton",
    );
    await page.locator('[aria-label="Show people"]').check();

    const scrub = async (frame) => page.locator("#frame").evaluate((input, value) => {
      input.value = String(value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }, frame);
    await scrub(15);
    await page.waitForFunction(() => Math.abs(document.querySelector("#video").currentTime - 0.5) < 0.04);
    assert.match(await page.locator("#detail").innerText(), /confirmed absent/);
    await scrub(30);
    assert.match(await page.locator("#detail").innerText(), /missing observation/);
    await scrub(31);
    assert.match(await page.locator("#detail").innerText(), /not sampled/);
    await page.selectOption("#entity", "shot-1/chair");
    await scrub(75);
    await page.waitForFunction(() => Math.abs(document.querySelector("#video").currentTime - 2.5) < 0.04);
    assert.match(await page.locator("#detail").innerText(), /low confidence/);
    assert.match(await page.locator("#detail").innerText(), /<chair>/, "entity labels are shown as text");
    assert.equal(await page.locator("#detail img, #detail script").count(), 0, "user label markup never becomes HTML");
    assert.equal(await page.locator("#shot").innerText(), "shot-1");
    assert.deepEqual(externalRequests, [], "page and assets make no off-origin requests");
    assert.deepEqual(errors, [], "viewer has no runtime errors");

    // A one-mask tint budget must evict without repeatedly reloading visible masks.
    const boundedPage = await browser.newPage();
    let maskLoads = 0;
    boundedPage.on("request", (request) => {
      if (request.url().includes("/environment/masks/")) maskLoads += 1;
    });
    await boundedPage.route("**/viewer.js", async (route) => {
      const response = await route.fetch();
      const script = (await response.text()).replace(
        "const MAX_TINT_BYTES = 64 * 1024 * 1024;",
        "const MAX_TINT_BYTES = 640 * 360 * 4;",
      );
      await route.fulfill({ response, body: script });
    });
    await boundedPage.goto(process.argv[2]);
    await boundedPage.waitForFunction(() => document.querySelector("#video").readyState >= 1);
    await boundedPage.waitForTimeout(250);
    await boundedPage.evaluate(() => {
      for (let i = 0; i < 20; i += 1) document.querySelector("#entity").dispatchEvent(new Event("change"));
    });
    await boundedPage.waitForTimeout(250);
    assert.equal(maskLoads, 3, "tint eviction must not trigger a visible-mask reload loop");
    await boundedPage.close();

    const missingMaskPage = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await missingMaskPage.route("**/environment/masks/floor.png", (route) =>
      route.fulfill({ status: 404, body: "missing fixture mask" }),
    );
    await missingMaskPage.goto(process.argv[2]);
    await missingMaskPage.waitForFunction(
      () => !document.querySelector("#error").hidden,
      null,
      { timeout: 2500 },
    );
    const maskError = await missingMaskPage.locator("#error").innerText();
    assert.match(maskError, /bundled.*mask.*could not be opened/i);
    await missingMaskPage.close();
    console.log(JSON.stringify({ saved, shot: "shot-1", externalRequests, errors }));
    console.log("PASS: offline video sync, shot transition, mask/skeleton geometry, family visibility and evidence states");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
