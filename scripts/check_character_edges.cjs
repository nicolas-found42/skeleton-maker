// SPDX-License-Identifier: MIT
// Run against scripts.make_character_fixture HTML with an installed Playwright module.
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
(async () => {
  const b = await chromium.launch({ headless: true });
  try {
    const p = await b.newPage({ viewport: { width: 640, height: 360 } });
    const errors = [];
    p.on("pageerror", (e) => errors.push(e.message));
    await p.goto(process.argv[2]);
    await p.waitForFunction(() => window.__stage);
    await p.evaluate(() => {
      window.__stage.current.playing = false;
      window.__stage.setFrame(10);
      window.__stage.render(true);
    });
    const failures = [];
    function check(ok, message) {
      if (!ok) failures.push(message);
    }
    await p.focus("#scrub");
    await p.keyboard.press("ArrowRight");
    let frame = await p.evaluate(() => window.__stage.current.frame);
    check(frame === 11, "focused ArrowRight steps once");
    await p.selectOption("#style", "clay");
    const collapse = await p.evaluate(() => {
      const v = window.__stage,
        r = [...v.rigs.values()].find((r) => r.group.visible),
        limb = r.parts.find((p) => p.caps);
      const k = v.current.frame - r.shot.global0 - r.track.start,
        n = v.stage.meta.joints.length,
        J = v.stage.J,
        d = r.track.data;
      const base = k * n * 3,
        a = base + J[limb.from] * 3,
        c = base + J[limb.to] * 3;
      const old = Array.from(d.slice(c, c + 3));
      for (let i = 0; i < 3; i++) d[c + i] = d[a + i];
      v.render(true);
      const visible = limb.caps.some((p) => p.visible);
      old.forEach((x, i) => (d[c + i] = x));
      return { visible };
    });
    check(!collapse.visible, "collapsed capsule hides ends");
    const missing = await p.evaluate(() => {
      const v = window.__stage,
        r = [...v.rigs.values()].find((r) => r.group.visible),
        k = v.current.frame - r.shot.global0 - r.track.start,
        n = v.stage.meta.joints.length,
        J = v.stage.J,
        d = r.track.data;
      const index = k * n * 3 + J.L_Shoulder * 3,
        old = d[index];
      d[index] = -32768;
      v.render(true);
      const visible = r.group.visible;
      d[index] = old;
      v.render(true);
      return { visible };
    });
    check(missing.visible, "core track remains visible with missing shoulder");
    await p.selectOption("#style", "mannequin");
    const owned = await p.evaluate(() => {
      const r = [...window.__stage.rigs.values()].find((r) => r.group.visible);
      window.disposedTapers = 0;
      const taper = r.parts.filter((p) => p.kind === "limb" && p.taper !== 1);
      taper.forEach((p) =>
        p.mesh.geometry.addEventListener(
          "dispose",
          () => window.disposedTapers++,
        ),
      );
      return taper.length;
    });
    await p.selectOption("#style", "robot");
    const disposed = await p.evaluate(() => window.disposedTapers);
    check(
      owned > 0 && disposed === owned,
      "recast disposes owned taper geometries",
    );
    await p.evaluate(() => {
      window.MediaRecorder = undefined;
    });
    await p.click("#rec");
    const record = await p.locator("#rec").textContent();
    check(/unavailable/i.test(record), "missing recorder shows unavailable");
    check(errors.length === 0, "no runtime errors");
    console.log(
      JSON.stringify(
        { frame, collapse, missing, owned, disposed, record, errors, failures },
        null,
        2,
      ),
    );
    assert.deepEqual(failures, []);
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
