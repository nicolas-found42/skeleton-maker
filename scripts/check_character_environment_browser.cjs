// SPDX-License-Identifier: MIT
// Browser proof for opt-in exact-frame registered geometry.
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
    const origin = new URL(process.argv[2]).origin;
    const screenshot = process.argv[3];
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
    const expected = await page.evaluate(async (url) => (await fetch(url)).json(), process.argv[4]);
    await page.waitForFunction(() => window.__stage?.environment);
    const frameMap = await page.evaluate(() => [0, 1, 20, 21].map((frame) => window.__stage.environment.sourceFrameFor(frame)));
    assert.deepEqual(frameMap, [0, 1, 90, 91], "shot and source-frame mapping is exact across the cut");

    const inspectFrame = async (globalFrame, sourceFrame) => page.evaluate(async ({ globalFrame, sourceFrame }) => {
      const view = window.__stage;
      view.current.playing = false;
      view.setFrame(globalFrame);
      view.render(true);
      const cloud = view.environment.clouds.get(sourceFrame);
      const points = cloud?.geometry.getAttribute("position");
      const knownPoint = points ? Array.from(points.array.slice(0, 3)) : null;
      let coloredPixels = 0;
      if (knownPoint) {
        const canvas = document.querySelector("#stage canvas");
        const image = await createImageBitmap(await (await fetch(canvas.toDataURL())).blob());
        const pixels = document.createElement("canvas");
        pixels.width = canvas.width;
        pixels.height = canvas.height;
        const context = pixels.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0);
        const projected = new view.camera.position.constructor(...knownPoint).project(view.camera);
        const centerX = Math.round((projected.x * 0.5 + 0.5) * canvas.width);
        const centerY = Math.round((-projected.y * 0.5 + 0.5) * canvas.height);
        for (let dy = -14; dy <= 14; dy++) {
          for (let dx = -14; dx <= 14; dx++) {
            const x = centerX + dx, y = centerY + dy;
            if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) continue;
            const [red, green, blue] = context.getImageData(x, y, 1, 1).data;
            if (green > 110 && green > red * 1.35 && green > blue * 1.05) coloredPixels++;
          }
        }
        image.close();
      }
      return {
        sourceFrame: view.environment.sourceFrameFor(globalFrame),
        cloudVisible: Boolean(cloud?.visible),
        knownPoint,
        status: document.querySelector("#environment-status").textContent,
        coloredPixels,
      };
    }, { globalFrame, sourceFrame });

    const first = await inspectFrame(0, 0);
    await page.screenshot({ path: screenshot });
    assert.equal(first.sourceFrame, 0);
    assert.equal(first.cloudVisible, true);
    const matchesExpected = (actual, point) => actual?.length === 3 && actual.every((value, axis) => Math.abs(value - point[axis]) <= 0.0001);
    assert(matchesExpected(first.knownPoint, expected["0"]), "frame 0 point equals independently transformed fixture geometry");
    assert.match(first.status, /Registered geometry · source frame 0/);
    assert(first.coloredPixels > 0, "known registered geometry contributes colored pixels to the rendered stage");

    const missing = await inspectFrame(1, 1);
    assert.equal(missing.cloudVisible, false, "a sampled frame without exact registration hides the previous cloud");
    assert.match(missing.status, /no exact registered camera transform for source frame 1/);

    const afterCut = await inspectFrame(20, 90);
    assert.equal(afterCut.sourceFrame, 90);
    assert.equal(afterCut.cloudVisible, true);
    assert(matchesExpected(afterCut.knownPoint, expected["90"]), "second shot uses its own camera and stage transforms");
    assert.notDeepEqual(afterCut.knownPoint, first.knownPoint, "shot-specific transforms change the displayed world point");
    assert(afterCut.coloredPixels > 0, "registered geometry remains rendered after the stage cut");
    const missingAfterCut = await inspectFrame(21, 91);
    assert.equal(missingAfterCut.cloudVisible, false);
    assert.match(missingAfterCut.status, /no exact registered camera transform for source frame 91/);
    assert.equal(errors.length, 0, "no browser runtime errors");
    assert.deepEqual(externalRequests, [], "browser makes no external requests");
    console.log(JSON.stringify({ first, missing, afterCut, missingAfterCut, frameMap, screenshot, errors }));
    console.log("PASS: exact registered point clouds render only on matching source frames across a shot cut");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
