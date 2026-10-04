// SPDX-License-Identifier: MIT
// Character stage viewer. Inlined into the generated HTML by skeleton_maker/character.py.
// Expects in scope: THREE, PAYLOAD (base64 gzip), SPECS (name -> spec), OPTIONS.
//
// Frame convention for props and limbs: x = person's LEFT, y = up, z = forward (right-handed).

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const MISSING = -32768;

// ---- data ------------------------------------------------------------------
async function gunzip(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}

function parseStage(buf) {
  const metaLen = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, metaLen)));
  const base = 4 + metaLen;
  const J = {};
  meta.joints.forEach((n, i) => (J[n] = i));
  let total = 0;
  for (const s of meta.shots) {
    s.global0 = total;
    total += s.n;
    for (const t of s.tracks) t.data = new Int16Array(buf.slice(base + t.offset, base + t.offset + t.n * meta.joints.length * 6));
  }
  return { meta, J, total };
}

// ---- hash / colour ---------------------------------------------------------
const hashId = (id) => ((id * 2654435761) >>> 0) % 997;

// ---- shared geometry -------------------------------------------------------
const G = {
  sphere: new THREE.SphereGeometry(1, 24, 16),
  box: new THREE.BoxGeometry(1, 1, 1),
  cone: new THREE.ConeGeometry(1, 1, 20),
  cylinder: new THREE.CylinderGeometry(1, 1, 1, 18),
  icosa: new THREE.IcosahedronGeometry(1, 1),
  torus: new THREE.TorusGeometry(1, 0.25, 8, 24),
};
G.cone.translate(0, 0.5, 0); // base at origin, tip up: pos is the base centre

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3();
function basisQuat(x, y, z, out) {
  _m.makeBasis(x, y, z);
  return out.setFromRotationMatrix(_m);
}

// ---- character rig ---------------------------------------------------------
function makeMaterial(def, color) {
  const o = { color: new THREE.Color(color) };
  let m;
  if (def.type === "basic") {
    m = new THREE.MeshBasicMaterial({ ...o, toneMapped: false });
  } else {
    m = new THREE.MeshStandardMaterial({ ...o, roughness: def.roughness ?? 0.7, metalness: def.metalness ?? 0 });
  }
  if (def.wireframe) m.wireframe = true;
  if (def.opacity !== undefined) { m.transparent = true; m.opacity = def.opacity; m.depthWrite = false; }
  return m;
}

const asList = (v) => (Array.isArray(v) ? v : [v]);

class Rig {
  constructor(spec, palette, track, shotRef, stage) {
    this.spec = spec; this.track = track; this.stage = stage; this.shot = shotRef;
    this.group = new THREE.Group();
    this.parts = [];
    this.chains = [];
    this.trails = [];
    this.mats = {};
    for (const [name, def] of Object.entries(spec.materials)) this.mats[name] = makeMaterial(def, palette[name] ?? "#cccccc");
    const J = stage.J;
    const wantJoint = (n) => { if (!(n in J)) throw new Error(`character "${spec.name}" uses unknown joint "${n}"`); return n; };
    for (const p of spec.parts) {
      if (p.type === "limb") {
        const from = asList(p.from), to = asList(p.to), n = Math.max(from.length, to.length);
        for (let i = 0; i < n; i++) this.addLimb(p, wantJoint(from[Math.min(i, from.length - 1)]), wantJoint(to[Math.min(i, to.length - 1)]));
      } else if (p.type === "prop") {
        for (const j of asList(p.joint)) this.addProp(p, wantJoint(j));
      } else if (p.type === "chain") {
        this.addChain(p);
      }
    }
    if (spec.trails) this.addTrails(spec.trails);
    this.group.visible = false;
  }
  add(mesh) { mesh.castShadow = true; mesh.receiveShadow = true; this.group.add(mesh); return mesh; }
  mat(name) { return this.mats[name]; }
  addLimb(p, from, to) {
    const shape = p.shape ?? "capsule";
    const geo = shape === "box" ? G.box : G.cylinder;
    const r1 = p.r ?? null, r2 = p.r2 ?? r1;
    const hx = p.w !== undefined ? p.w / 2 : r1, hz = p.d !== undefined ? p.d / 2 : r1;
    const taper = r1 && r2 ? r2 / r1 : 1;
    let mesh;
    if (shape === "box") mesh = this.add(new THREE.Mesh(G.box, this.mat(p.mat)));
    else {
      const g = taper === 1 ? G.cylinder : new THREE.CylinderGeometry(taper, 1, 1, 18);
      mesh = this.add(new THREE.Mesh(g, this.mat(p.mat)));
    }
    const caps = shape === "capsule" ? [this.add(new THREE.Mesh(G.sphere, this.mat(p.mat))), this.add(new THREE.Mesh(G.sphere, this.mat(p.mat)))] : null;
    this.parts.push({ kind: "limb", from, to, mesh, caps, shape, hx, hz, taper, box: shape === "box", ext: p.extend ?? [0, 0] });
  }
  addProp(p, joint) {
    const g = G[p.shape] ?? G.sphere;
    const mesh = this.add(new THREE.Mesh(g, this.mat(p.mat)));
    const rot = p.rot ? new THREE.Quaternion().setFromEuler(new THREE.Euler(...p.rot.map((d) => (d * Math.PI) / 180))) : null;
    this.parts.push({ kind: "prop", joint, mesh, frame: p.frame ?? "body", pos: p.pos ?? [0, 0, 0], size: p.size ?? [0.05, 0.05, 0.05], rot });
  }
  addChain(p) {
    const segs = [];
    const [r0, r1] = p.r;
    for (let i = 0; i < p.n; i++) {
      const t = i / Math.max(1, p.n - 1);
      const mat = i === p.n - 1 && p.tip ? this.mat(p.tip) : this.mat(p.mat);
      segs.push({ mesh: this.add(new THREE.Mesh(G.sphere, mat)), r: r0 + (r1 - r0) * t, p: new THREE.Vector3(), init: false });
    }
    this.chains.push({ spec: p, segs });
  }
  addTrails(t) {
    for (const j of t.joints) {
      const N = t.length;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      const base = this.mat(t.mat).color;
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, toneMapped: false }));
      line.frustumCulled = false;
      this.group.add(line);
      this.trails.push({ joint: j, N, line, base, hist: [] });
    }
  }
  dispose() {
    for (const m of Object.values(this.mats)) m.dispose();
    const shared = new Set(Object.values(G));
    this.group.traverse((o) => {
      if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose();
      if (o.isLine) o.material.dispose();
    });
  }
}

// ---- pose evaluation -------------------------------------------------------
const pos = {}; // joint name -> Vector3 (reused), valid flags in `ok`
const ok = {};
function loadFrame(stage, shot, track, k, out, outOk) {
  const nj = stage.meta.joints.length, c = shot.center, d = track.data, base = k * nj * 3;
  for (let j = 0; j < nj; j++) {
    const name = stage.meta.joints[j];
    const x = d[base + j * 3];
    const v = out[name] || (out[name] = new THREE.Vector3());
    if (x === MISSING || d[base + j * 3 + 1] === MISSING) { outOk[name] = false; continue; }
    v.set(x / 1000 + c[0], d[base + j * 3 + 1] / 1000 + c[1], d[base + j * 3 + 2] / 1000 + c[2]);
    outOk[name] = true;
  }
}

const F = { up: new THREE.Vector3(), left: new THREE.Vector3(), fwd: new THREE.Vector3(), hup: new THREE.Vector3(), hleft: new THREE.Vector3(), hfwd: new THREE.Vector3() };
function computeFrames() {
  const up = F.up.copy(pos.Chest).sub(pos.Hips); if (up.lengthSq() < 1e-6) up.set(0, 1, 0); up.normalize();
  const left = F.left;
  if (ok.L_Shoulder && ok.R_Shoulder) left.copy(pos.L_Shoulder).sub(pos.R_Shoulder);
  else left.set(1, 0, 0);
  left.addScaledVector(up, -left.dot(up));
  if (left.lengthSq() < 1e-6) { left.set(0, 0, 1); left.addScaledVector(up, -left.dot(up)); }
  left.normalize();
  F.fwd.crossVectors(left, up).normalize(); // x=left, y=up -> z = x cross y
  // head: lateral from the head-side markers, up from head-top, falls back to the body
  const hu = F.hup.copy(pos.HeadTop).sub(pos.Head); if (!ok.HeadTop || hu.lengthSq() < 1e-6) hu.copy(up); hu.normalize();
  const hl = F.hleft.copy(pos.L_HeadSide).sub(pos.R_HeadSide);
  if (!ok.L_HeadSide || !ok.R_HeadSide || hl.lengthSq() < 1e-6) hl.copy(left);
  hl.addScaledVector(hu, -hl.dot(hu)); hl.normalize();
  F.hfwd.crossVectors(hl, hu).normalize();
}

const _t = new THREE.Vector3(), _col = new THREE.Color();
function applyRig(rig, k, time) {
  const stage = rig.stage, shot = rig.shot, track = rig.track;
  loadFrame(stage, shot, track, k, pos, ok);
  if (!(ok.Hips && ok.Chest && ok.Head)) { rig.group.visible = false; return; }
  rig.group.visible = true;
  computeFrames();
  const u = track.unit * (OPTIONS.scale ?? 1);
  // fade in/out where a person appears or leaves mid-shot, not at the shot's own first and last frame
  const inEdge = track.start > 0 ? k : Infinity;
  const outEdge = track.start + track.n < shot.n ? track.n - 1 - k : Infinity;
  const edge = Math.min(inEdge, outEdge);
  const fade = clamp(edge / 6, 0, 1), s = fade * fade * (3 - 2 * fade);
  rig.fade = Math.max(s, 0.001); // fade by scaling each part about its own centre, not the group
  for (const p of rig.parts) {
    if (p.kind === "limb") {
      if (!ok[p.from] || !ok[p.to]) { p.mesh.visible = false; if (p.caps) p.caps.forEach((c) => (c.visible = false)); continue; }
      p.mesh.visible = true;
      const a = _a.copy(pos[p.from]), b = _b.copy(pos[p.to]);
      const dir = _c.copy(b).sub(a); const len = dir.length();
      if (len < 1e-5) { p.mesh.visible = false; if (p.caps) p.caps.forEach((c) => (c.visible = false)); continue; }
      dir.divideScalar(len);
      a.addScaledVector(dir, -p.ext[0] * u); b.addScaledVector(dir, p.ext[1] * u);
      const L = a.distanceTo(b);
      const x = _t.copy(F.left); x.addScaledVector(dir, -x.dot(dir));
      if (x.lengthSq() < 1e-6) x.copy(F.fwd).addScaledVector(dir, -F.fwd.dot(dir));
      x.normalize();
      const z = new THREE.Vector3().crossVectors(x, dir);
      basisQuat(x, dir, z, p.mesh.quaternion);
      p.mesh.position.copy(a).add(b).multiplyScalar(0.5);
      const f = rig.fade;
      if (p.box) p.mesh.scale.set(p.hx * 2 * u * f, L * f, p.hz * 2 * u * f);
      else p.mesh.scale.set(p.hx * u * f, L * f, p.hz * u * f);
      if (p.caps) {
        p.caps[0].visible = p.caps[1].visible = true;
        p.caps[0].position.copy(a); p.caps[1].position.copy(b);
        p.caps[0].scale.set(p.hx * u * f, Math.min(p.hx, p.hz) * u * f, p.hz * u * f);
        const t2 = p.taper;
        p.caps[1].scale.set(p.hx * t2 * u * f, Math.min(p.hx, p.hz) * t2 * u * f, p.hz * t2 * u * f);
        p.caps[0].quaternion.copy(p.mesh.quaternion); p.caps[1].quaternion.copy(p.mesh.quaternion);
      }
    } else if (p.kind === "prop") {
      if (!ok[p.joint]) { p.mesh.visible = false; continue; }
      p.mesh.visible = true;
      const head = p.frame === "head";
      const l = head ? F.hleft : F.left, up = head ? F.hup : F.up, fw = head ? F.hfwd : F.fwd;
      p.mesh.position.copy(pos[p.joint])
        .addScaledVector(l, p.pos[0] * u).addScaledVector(up, p.pos[1] * u).addScaledVector(fw, p.pos[2] * u);
      basisQuat(l, up, fw, p.mesh.quaternion);
      if (p.rot) p.mesh.quaternion.multiply(p.rot);
      const f = rig.fade;
      p.mesh.scale.set(p.size[0] * u * f, p.size[1] * u * f, p.size[2] * u * f);
    }
  }
  for (const ch of rig.chains) {
    const sp = ch.spec, a = pos[sp.joint];
    if (!ok[sp.joint]) { ch.segs.forEach((sg) => { sg.mesh.visible = false; sg.init = false; }); continue; }
    ch.segs.forEach((sg) => { sg.mesh.visible = true; });
    const dirw = _t.set(0, 0, 0).addScaledVector(F.left, sp.dir[0]).addScaledVector(F.up, sp.dir[1]).addScaledVector(F.fwd, sp.dir[2]).normalize();
    let prev = a.clone();
    ch.segs.forEach((sg, i) => {
      const rest = prev.clone().addScaledVector(dirw, sp.len * u).add(new THREE.Vector3(0, -0.01 * i * u, 0));
      if (!sg.init) { sg.p.copy(rest); sg.init = true; }
      sg.p.lerp(rest, 1 - sp.lag);
      const d = sg.p.clone().sub(prev); const dl = d.length() || 1;
      sg.p.copy(prev).addScaledVector(d, (sp.len * u) / dl);
      sg.mesh.position.copy(sg.p);
      sg.mesh.scale.setScalar(sg.r * u * rig.fade);
      prev = sg.p;
    });
  }
  for (const tr of rig.trails) {
    if (!ok[tr.joint]) continue;
    tr.hist.unshift(pos[tr.joint].clone()); if (tr.hist.length > tr.N) tr.hist.pop();
    const P = tr.line.geometry.attributes.position, C = tr.line.geometry.attributes.color;
    for (let i = 0; i < tr.N; i++) {
      const h = tr.hist[Math.min(i, tr.hist.length - 1)];
      P.setXYZ(i, h.x, h.y, h.z);
      const f = i < tr.hist.length ? Math.pow(1 - i / tr.N, 2) * rig.fade : 0;
      C.setXYZ(i, tr.base.r * f, tr.base.g * f, tr.base.b * f);
    }
    P.needsUpdate = C.needsUpdate = true;
  }
}

// ---- app -------------------------------------------------------------------
let stage, renderer, scene, camera, sun, floor, grid, clock;
const rigs = new Map(); // `${shot}:${track}` -> Rig
let current = { frame: 0, playing: true, speed: 1 };
let viewMode = "original";
const orbit = { target: new THREE.Vector3(), theta: 0, phi: 1.2, radius: 10 };
let styleChoice = OPTIONS.character || "auto";
const castOverride = {}; // track id -> style name
const styleNames = Object.keys(SPECS);
let activeShot = -1;
let showLabels = false, labels = {};

function pickStyle(track) { return castOverride[track.id] || (styleChoice !== "auto" ? styleChoice : styleNames[hashId(track.id) % styleNames.length]); }

function rigFor(si, ti) {
  const key = `${si}:${ti}`;
  let rig = rigs.get(key);
  const track = stage.meta.shots[si].tracks[ti];
  const name = pickStyle(track);
  if (rig && rig.spec.name === name) return rig;
  if (rig) { scene.remove(rig.group); rig.dispose(); }
  const spec = SPECS[name];
  const palette = spec.palettes[hashId(track.id + 7) % spec.palettes.length];
  rig = new Rig(spec, palette, track, stage.meta.shots[si], stage);
  scene.add(rig.group);
  rigs.set(key, rig);
  return rig;
}

function shotAt(g) {
  const shots = stage.meta.shots;
  for (let i = shots.length - 1; i >= 0; i--) if (g >= shots[i].global0) return i;
  return 0;
}

function frameCamera(si) {
  const shot = stage.meta.shots[si];
  const c = shot.camera;
  if (viewMode === "original") {
    camera.position.set(...c.position);
    camera.lookAt(camera.position.clone().add(new THREE.Vector3(...c.direction).multiplyScalar(10)));
  } else {
    orbit.target.set(shot.center[0], 0.9, shot.center[2]);
    orbitApply();
  }
  const R = 40;
  sun.position.set(shot.center[0] + 6, 14, shot.center[2] + 8);
  sun.target.position.set(shot.center[0], 0, shot.center[2]);
  sun.shadow.camera.left = -R; sun.shadow.camera.right = R; sun.shadow.camera.top = R; sun.shadow.camera.bottom = -R;
  sun.shadow.camera.updateProjectionMatrix();
  floor.position.set(shot.center[0], -0.002, shot.center[2]); grid.position.set(shot.center[0], 0, shot.center[2]);
  floor.visible = shot.floor_source !== "camera_origin";
  grid.visible = floor.visible && $("gridbox").checked;
}

function orbitApply() {
  const o = orbit;
  camera.position.set(
    o.target.x + o.radius * Math.sin(o.phi) * Math.sin(o.theta),
    o.target.y + o.radius * Math.cos(o.phi),
    o.target.z + o.radius * Math.sin(o.phi) * Math.cos(o.theta));
  camera.lookAt(o.target);
}
function enterOrbit() {
  if (viewMode === "orbit") return;
  viewMode = "orbit";
  const shot = stage.meta.shots[Math.max(0, activeShot)];
  orbit.target.set(shot.center[0], 0.9, shot.center[2]);
  const off = camera.position.clone().sub(orbit.target);
  orbit.radius = clamp(off.length(), 4, 80);
  orbit.theta = Math.atan2(off.x, off.z);
  orbit.phi = clamp(Math.acos(clamp(off.y / orbit.radius, -1, 1)), 0.2, 1.5);
  $("view-orbit").classList.add("on"); $("view-original").classList.remove("on");
  orbitApply();
}
function enterOriginal() {
  viewMode = "original";
  $("view-original").classList.add("on"); $("view-orbit").classList.remove("on");
  if (activeShot >= 0) frameCamera(activeShot);
}

function buildStyleUI() {
  const sel = $("style");
  sel.innerHTML = "";
  for (const n of ["auto", ...styleNames]) { const o = document.createElement("option"); o.value = n; o.textContent = n === "auto" ? "auto-cast (a different one per person)" : n; sel.appendChild(o); }
  sel.value = styleChoice;
  sel.onchange = () => { styleChoice = sel.value; for (const k of Object.keys(castOverride)) delete castOverride[k]; rebuildAll(); if (activeShot >= 0) buildCastUI(activeShot); };
}
function rebuildAll() { for (const r of rigs.values()) { scene.remove(r.group); r.dispose(); } rigs.clear(); render(true); }

function setFrame(g) {
  current.frame = clamp(g, 0, stage.total - 1);
  $("scrub").value = current.frame;
}

const _v = new THREE.Vector3();
function updateLabels(si) {
  const shot = stage.meta.shots[si];
  for (const k of Object.keys(labels)) { labels[k].style.display = "none"; }
  if (!showLabels) return;
  const k = current.frame - shot.global0;
  shot.tracks.forEach((t, ti) => {
    const i = k - t.start; if (i < 0 || i >= t.n) return;
    const rig = rigs.get(`${si}:${ti}`); if (!rig || !rig.group.visible) return;
    loadFrame(stage, shot, t, i, pos, ok); if (!ok.HeadTop) return;
    _v.copy(pos.HeadTop).add(new THREE.Vector3(0, 0.25 * t.unit, 0)).project(camera);
    if (_v.z > 1) return;
    let el = labels[`${si}:${ti}`];
    if (!el) { el = document.createElement("div"); el.className = "label"; el.textContent = `#${t.id}`; $("labels").appendChild(el); labels[`${si}:${ti}`] = el; }
    el.style.display = "block";
    el.style.transform = `translate(-50%,-100%) translate(${(_v.x * 0.5 + 0.5) * innerWidth}px,${(-_v.y * 0.5 + 0.5) * innerHeight}px)`;
  });
}

function render(force) {
  const si = shotAt(current.frame);
  if (si !== activeShot) {
    activeShot = si;
    for (const [key, r] of rigs) r.group.visible = false;
    frameCamera(si);
    $("shot").textContent = `shot ${si + 1} / ${stage.meta.shots.length}`;
    buildCastUI(si);
  }
  const shot = stage.meta.shots[si];
  const k = current.frame - shot.global0;
  let visible = 0, followed = false;
  shot.tracks.forEach((t, ti) => {
    const i = k - t.start;
    const key = `${si}:${ti}`;
    if (i < 0 || i >= t.n) { const r = rigs.get(key); if (r) r.group.visible = false; return; }
    const rig = rigFor(si, ti);
    applyRig(rig, i, current.frame);
    if (rig.group.visible) visible++;
    if (followId !== null && t.id === followId && rig.group.visible && ok.Hips) { _f.copy(pos.Hips); followed = true; }
  });
  if (followed && viewMode === "orbit") { orbit.target.lerp(_f, force ? 1 : 0.2); orbitApply(); }
  $("count").textContent = `${visible} ${visible === 1 ? "person" : "people"}`;
  const fps = stage.meta.fps;
  $("time").textContent = `${(current.frame / fps).toFixed(1)}s / ${(stage.total / fps).toFixed(1)}s`;
  syncVideo();
  updateLabels(si);
  renderer.render(scene, camera);
}

// Follow: keep the orbit camera on one person, e.g. the performer in a crowd.
let followId = null;
const _f = new THREE.Vector3();
function buildFollowUI(si) {
  const sel = $("follow");
  sel.innerHTML = "";
  const none = document.createElement("option"); none.value = ""; none.textContent = "nobody"; sel.appendChild(none);
  const seen = new Set();
  for (const t of stage.meta.shots[si].tracks) {
    if (seen.has(t.id)) continue; seen.add(t.id);
    const o = document.createElement("option"); o.value = String(t.id); o.textContent = `#${t.id}`; sel.appendChild(o);
  }
  sel.value = followId !== null && seen.has(followId) ? String(followId) : "";
  if (sel.value === "") followId = null;
  sel.onchange = () => {
    followId = sel.value === "" ? null : Number(sel.value);
    if (followId !== null) { enterOrbit(); orbit.radius = Math.min(orbit.radius, 6); render(true); }
  };
}

function buildCastUI(si) {
  buildFollowUI(si);
  const box = $("cast"); box.innerHTML = "";
  const shot = stage.meta.shots[si];
  const seen = new Set();
  for (const t of shot.tracks) {
    if (seen.has(t.id)) continue; seen.add(t.id);
    const row = document.createElement("label"); row.className = "castrow";
    row.innerHTML = `<span>#${t.id}</span>`;
    const s = document.createElement("select");
    for (const n of styleNames) { const o = document.createElement("option"); o.value = n; o.textContent = n; s.appendChild(o); }
    s.value = pickStyle(t);
    s.onchange = () => { castOverride[t.id] = s.value; for (const [k, r] of rigs) if (r.track.id === t.id) { scene.remove(r.group); r.dispose(); rigs.delete(k); } };
    row.appendChild(s); box.appendChild(row);
  }
}

// ---- video picture-in-picture ---------------------------------------------
let video = null;
function initVideo() {
  if (!OPTIONS.video) return;
  video = $("video"); video.src = OPTIONS.video; video.muted = true; video.style.display = "block";
  video.addEventListener("error", () => { video.style.display = "none"; video = null; });
}
function syncVideo() {
  if (!video) return;
  const shot = stage.meta.shots[shotAt(current.frame)];
  const t = (shot.frame0 + current.frame - shot.global0) / stage.meta.fps;
  video.playbackRate = current.speed;
  if (Math.abs(video.currentTime - t) > 0.12) video.currentTime = t;
  if (current.playing && video.paused) video.play().catch(() => {});
  if (!current.playing && !video.paused) video.pause();
}

// ---- boot ------------------------------------------------------------------
async function boot() {
  stage = parseStage(await gunzip(PAYLOAD));
  $("title").textContent = OPTIONS.title || "skeleton stage";
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor(0x0e1016);
  $("stage").appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0e1016, 40, 140);
  camera = new THREE.PerspectiveCamera(OPTIONS.fov || 45, 1, 0.1, 400);
  scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x20222c, 1.1));
  sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 80;
  scene.add(sun, sun.target);
  floor = new THREE.Mesh(new THREE.CircleGeometry(80, 64), new THREE.MeshStandardMaterial({ color: 0x1a1d27, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  grid = new THREE.GridHelper(160, 160, 0x34384a, 0x232636); grid.material.transparent = true; grid.material.opacity = 0.5; scene.add(grid);

  const scrub = $("scrub"); scrub.max = stage.total - 1; scrub.value = 0;
  scrub.oninput = () => { current.frame = +scrub.value; render(); };
  const ticks = $("ticks"); ticks.innerHTML = "";
  for (const s of stage.meta.shots.slice(1)) { const d = document.createElement("i"); d.style.left = `${(s.global0 / (stage.total - 1)) * 100}%`; ticks.appendChild(d); }
  $("play").onclick = () => { current.playing = !current.playing; $("play").textContent = current.playing ? "❚❚" : "▶"; };
  $("speed").onchange = (e) => (current.speed = +e.target.value);
  $("view-original").onclick = enterOriginal; $("view-orbit").onclick = enterOrbit;
  $("shadows").onchange = (e) => { sun.castShadow = e.target.checked; };
  $("gridbox").onchange = (e) => { grid.visible = floor.visible && e.target.checked; };
  $("labelbox").onchange = (e) => { showLabels = e.target.checked; };
  $("shot-png").onclick = () => { renderer.render(scene, camera); const a = document.createElement("a"); a.href = renderer.domElement.toDataURL("image/png"); a.download = "skeleton-stage.png"; a.click(); };
  $("rec").onclick = toggleRecord;
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range") return;
    if (e.code === "Space") { e.preventDefault(); $("play").click(); }
    if (e.code === "ArrowRight") { e.preventDefault(); current.playing = false; setFrame(current.frame + (e.shiftKey ? 10 : 1)); render(); }
    if (e.code === "ArrowLeft") { e.preventDefault(); current.playing = false; setFrame(current.frame - (e.shiftKey ? 10 : 1)); render(); }
  });
  setupPointer();
  buildStyleUI(); initVideo();
  addEventListener("resize", resize); resize();
  clock = new THREE.Clock();
  let acc = 0;
  const loop = () => {
    requestAnimationFrame(loop);
    const dt = clock.getDelta();
    if (current.playing) {
      acc += dt * stage.meta.fps * current.speed;
      const step = Math.floor(acc); acc -= step;
      if (step) { let g = current.frame + step; if (g >= stage.total) g = 0; setFrame(g); }
    }
    render();
  };
  $("loading").style.display = "none";
  loop();
  window.__stage = { stage, rigs, current, get camera() { return camera; }, setFrame, render, enterOrbit, enterOriginal, orbit };
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}

function setupPointer() {
  const el = renderer.domElement; let drag = null;
  el.addEventListener("pointerdown", (e) => { enterOrbit(); drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointerup", () => (drag = null));
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  el.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), upv = new THREE.Vector3(0, 1, 0);
      orbit.target.addScaledVector(right, -dx * orbit.radius * 0.0015).addScaledVector(upv, dy * orbit.radius * 0.0015);
    } else { orbit.theta -= dx * 0.006; orbit.phi = clamp(orbit.phi - dy * 0.006, 0.15, 1.52); }
    orbitApply();
  });
  el.addEventListener("wheel", (e) => { e.preventDefault(); enterOrbit(); orbit.radius = clamp(orbit.radius * Math.exp(e.deltaY * 0.001), 1.5, 120); orbitApply(); }, { passive: false });
}

// ---- recording -------------------------------------------------------------
let recorder = null, chunks = [];
function toggleRecord() {
  const btn = $("rec");
  if (recorder) { recorder.stop(); return; }
  const unavailable = () => { btn.textContent = "Recording unavailable"; btn.title = "Use a browser supporting canvas capture and WebM recording."; btn.disabled = true; };
  if (typeof MediaRecorder === "undefined" || !renderer.domElement.captureStream) { unavailable(); return; }
  const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  if (!mime) { unavailable(); return; }
  let stream;
  try {
    stream = renderer.domElement.captureStream(stage.meta.fps);
    recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
  } catch (_) { if (stream) stream.getTracks().forEach((t) => t.stop()); recorder = null; unavailable(); return; }
  chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" })); a.download = "skeleton-stage.webm"; a.click();
    recorder = null; btn.textContent = "● Record"; btn.classList.remove("on");
    stream.getTracks().forEach((t) => t.stop());
  };
  recorder.onerror = () => { stream.getTracks().forEach((t) => t.stop()); recorder = null; btn.classList.remove("on"); unavailable(); };
  setFrame(0); current.playing = true; $("play").textContent = "❚❚";
  try { recorder.start(); btn.textContent = "■ Stop & save"; btn.classList.add("on"); }
  catch (_) { stream.getTracks().forEach((t) => t.stop()); recorder = null; unavailable(); }
}

boot().catch((err) => { $("loading").textContent = "Could not load the stage: " + err.message; console.error(err); });
