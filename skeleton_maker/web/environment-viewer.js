// Offline source-frame viewer. User-derived values are written with textContent.
(() => {
  "use strict";
  const payload = JSON.parse(document.getElementById("payload").textContent);
  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const frameInput = document.getElementById("frame");
  const frameLabel = document.getElementById("frame-label");
  const detail = document.getElementById("detail");
  const entitySelect = document.getElementById("entity");
  const families = new Map();
  const observations = new Map();
  const poses = new Map(payload.poses.map((record) => [record.frame_id, record.detections]));
  const sampled = new Set(payload.processed_frames.map((record) => record.frame_id));
  const colors = { surface: "#34d399", object: "#fbbf24", vehicle: "#60a5fa", person: "#f472b6" };
  const images = new Map();
  const tintedMasks = new Map();
  const MAX_TINT_BYTES = 64 * 1024 * 1024;
  let tintBytes = 0;
  let currentFrame = 0;

  const familyLabels = { person: "people", surface: "surfaces", object: "objects", vehicle: "vehicles" };
  for (const family of ["person", "surface", "object", "vehicle"]) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.setAttribute("aria-label", `Show ${familyLabels[family]}`);
    families.set(family, input);
    input.addEventListener("change", paint);
    label.append(input, document.createTextNode(`${familyLabels[family][0].toUpperCase()}${familyLabels[family].slice(1)}`));
    document.querySelector(".families").append(label);
  }
  for (const observation of payload.observations) {
    if (!observations.has(observation.entity)) observations.set(observation.entity, []);
    observations.get(observation.entity).push(observation);
  }
  for (const item of payload.entities) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.labels.native} · ${item.id}`;
    entitySelect.append(option);
  }
  if (payload.entities.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No environment entities";
    entitySelect.append(option);
  }
  document.getElementById("geometry").textContent = `Geometry: ${payload.geometry.status} — ${payload.geometry.reason}`;
  frameInput.max = String(Math.max(0, payload.source.frame_count - 1));
  canvas.width = payload.source.width;
  canvas.height = payload.source.height;
  const fps = payload.source.frame_rate[0] / payload.source.frame_rate[1];
  const bundle = document.currentScript?.dataset.bundle || new URL(video.getAttribute("src"), location.href).pathname.split("/").slice(0, -1).join("/");
  // The asset references were validated during export; encode each path segment when loading it.
  function imageFor(rel) {
    if (!images.has(rel)) {
      const image = new Image();
      image.onload = () => { if (images.get(rel) === image) paint(); };
      image.onerror = () => {
        const error = document.getElementById("error");
        error.textContent = "A bundled mask image could not be opened. Restore the local viewer asset bundle beside this HTML file.";
        error.hidden = false;
      };
      image.src = `${bundle}/${rel.split("/").map(encodeURIComponent).join("/")}`;
      images.set(rel, image);
    }
    return images.get(rel);
  }
  function tintedMask(rel, color) {
    const key = JSON.stringify([rel, color]);
    if (tintedMasks.has(key)) {
      const tint = tintedMasks.get(key);
      tintedMasks.delete(key);
      tintedMasks.set(key, tint);
      return tint;
    }
    const image = imageFor(rel);
    if (!image.complete || !image.naturalWidth) return null;
    const tint = document.createElement("canvas");
    tint.width = canvas.width; tint.height = canvas.height;
    const tintCtx = tint.getContext("2d");
    tintCtx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const maskPixels = tintCtx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < maskPixels.data.length; i += 4) {
      const maskAlpha = maskPixels.data[i];
      maskPixels.data[i] = 255;
      maskPixels.data[i + 1] = 255;
      maskPixels.data[i + 2] = 255;
      maskPixels.data[i + 3] = maskAlpha;
    }
    tintCtx.putImageData(maskPixels, 0, 0);
    tintCtx.globalCompositeOperation = "source-in";
    tintCtx.fillStyle = color;
    tintCtx.fillRect(0, 0, canvas.width, canvas.height);
    const bytes = tint.width * tint.height * 4;
    if (bytes <= MAX_TINT_BYTES) {
      while (tintBytes + bytes > MAX_TINT_BYTES) {
        const oldest = tintedMasks.keys().next().value;
        const evicted = tintedMasks.get(oldest);
        tintBytes -= evicted.width * evicted.height * 4;
        tintedMasks.delete(oldest);
        evicted.width = 0; evicted.height = 0;
      }
      tintedMasks.set(key, tint);
      tintBytes += bytes;
    }
    return tint;
  }
  function frameStatus(entityId) {
    if (!sampled.has(currentFrame)) return { label: "not sampled", kind: "missing", observation: null };
    const obs = (observations.get(entityId) || []).find((item) => item.frame_id === currentFrame);
    if (!obs) return { label: "missing observation", kind: "missing", observation: null };
    if (obs.visibility === "absent") return { label: "confirmed absent", kind: "absent", observation: obs };
    if (obs.visibility === "uncertain") return { label: "low confidence · uncertain visibility", kind: "low", observation: obs };
    return { label: obs.visibility, kind: "observed", observation: obs };
  }
  function paintSkeletons() {
    if (!families.get("person").checked) return;
    const detections = poses.get(currentFrame) || [];
    for (const detection of detections) {
      const points = detection.keypoints_2d || [];
      const confidence = detection.keypoints_confidence || [];
      ctx.strokeStyle = colors.person;
      ctx.fillStyle = colors.person;
      ctx.lineWidth = Math.max(1.5, canvas.width / 500);
      for (const [child, parent] of payload.skeleton_links) {
        const a = points[parent], b = points[child];
        if (!a || !b || confidence[parent] <= 0 || confidence[child] <= 0) continue;
        if (![...a, ...b].every(Number.isFinite)) continue;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      for (let index = 0; index < points.length; index += 1) {
        const point = points[index];
        if (!point || confidence[index] <= 0 || !point.every(Number.isFinite)) continue;
        ctx.beginPath(); ctx.arc(point[0], point[1], Math.max(2, canvas.width / 320), 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  function paintEnvironment() {
    for (const entity of payload.entities) {
      const family = entity.family === "person" ? "person" : entity.family;
      if (!families.get(family)?.checked) continue;
      const status = frameStatus(entity.id);
      const observation = status.observation;
      if (!observation || observation.visibility === "absent") continue;
      ctx.save();
      ctx.strokeStyle = colors[family] || "#fff";
      ctx.fillStyle = colors[family] || "#fff";
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.setLineDash(observation.visibility === "uncertain" ? [8, 6] : []);
      if (observation.mask) {
        const tint = tintedMask(observation.mask.asset, colors[family] || "#fff");
        if (tint) {
          ctx.globalAlpha = observation.visibility === "uncertain" ? 0.22 : 0.3;
          ctx.drawImage(tint, 0, 0);
          ctx.globalAlpha = 1;
        }
      }
      if (observation.bbox) {
        const [x0, y0, x1, y1] = observation.bbox;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
      if (entity.id === entitySelect.value && observation.bbox) {
        const [x, y] = observation.bbox;
        ctx.font = `${Math.max(12, canvas.width / 70)}px system-ui`;
        ctx.fillText(entity.labels.native, x, Math.max(14, y - 4));
      }
      ctx.restore();
    }
  }
  function paint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paintEnvironment();
    paintSkeletons();
    updateDetail();
  }
  function updateDetail() {
    const id = entitySelect.value;
    const entity = payload.entities.find((item) => item.id === id);
    detail.replaceChildren();
    if (!entity) return;
    const status = frameStatus(id);
    const rows = [
      ["Label", entity.labels.native], ["Family", entity.family], ["Motion", entity.motion],
      ["Frame status", status.label], ["Visibility", status.observation?.visibility || "—"],
      ["Confidence", status.observation?.score === undefined ? "—" : `${status.observation.score} (${status.observation.score_meaning})`],
      ["Source frames", (observations.get(id) || []).map((item) => item.frame_id).join(", ") || "none"],
    ];
    for (const [term, value] of rows) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = term; dd.textContent = value;
      if (term === "Frame status") { dd.className = `status ${status.kind}`; }
      detail.append(dt, dd);
    }
  }
  function setFrame(frame, seek) {
    const nextFrame = Math.max(0, Math.min(Number(frame) || 0, payload.source.frame_count - 1));
    // Raw images serve repeated paints and tint eviction only for the current frame.
    if (nextFrame !== currentFrame) images.clear();
    currentFrame = nextFrame;
    frameInput.value = String(currentFrame);
    frameLabel.value = String(currentFrame);
    if (seek && Number.isFinite(fps) && fps > 0 && Math.abs(video.currentTime - currentFrame / fps) > 1 / fps / 2) {
      video.currentTime = currentFrame / fps;
    }
    const shot = payload.shots.find((item) => item.first_frame <= currentFrame && item.last_frame >= currentFrame);
    document.getElementById("shot").textContent = shot ? shot.id : "No shot metadata";
    paint();
  }
  frameInput.addEventListener("input", () => setFrame(frameInput.value, true));
  entitySelect.addEventListener("change", paint);
  video.addEventListener("timeupdate", () => setFrame(Math.round(video.currentTime * fps), false));
  video.addEventListener("error", () => {
    const error = document.getElementById("error");
    error.textContent = "The bundled source video could not be opened. Restore the local viewer asset bundle beside this HTML file.";
    error.hidden = false;
  });
  setFrame(0, false);
})();
