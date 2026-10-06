// This module fragment is inlined only when `character --environment` is used.
const environmentClouds = new Map();
let visibleEnvironmentCloud = null;
function drawEnvironment(shotIndex, shotFrame) {
  const shot = stage.meta.shots[shotIndex];
  const sourceFrame = shot.frame0 + shotFrame;
  const item = OPTIONS.environment.frames[String(sourceFrame)];
  const status = document.getElementById("environment-status");
  if (visibleEnvironmentCloud) visibleEnvironmentCloud.visible = false;
  visibleEnvironmentCloud = null;
  if (OPTIONS.environment.status !== "registered_metric") {
    status.textContent = `Environment geometry hidden: ${OPTIONS.environment.reason}`;
    return;
  }
  if (!item) {
    status.textContent = `Environment geometry hidden: no exact stage record for source frame ${sourceFrame}`;
    return;
  }
  if (!item.points.length) {
    status.textContent = `Environment geometry hidden: ${item.reason || "no registered point cloud"}`;
    return;
  }
  status.textContent = `Registered geometry · source frame ${sourceFrame} · ${item.points.length} points`;
  let cloud = environmentClouds.get(sourceFrame);
  if (!cloud) {
    const positions = new Float32Array(item.points.length * 3);
    item.points.forEach((point, index) => positions.set(point, index * 3));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0x51d6ad,
      size: 0.09,
      sizeAttenuation: true,
      depthWrite: false,
    });
    cloud = new THREE.Points(geometry, material);
    cloud.frustumCulled = false;
    scene.add(cloud);
    environmentClouds.set(sourceFrame, cloud);
  }
  cloud.visible = true;
  visibleEnvironmentCloud = cloud;
}
const environmentView = {
  clouds: environmentClouds,
  sourceFrameFor(globalFrame) {
    const shotIndex = shotAt(globalFrame);
    const shot = stage.meta.shots[shotIndex];
    return shot.frame0 + globalFrame - shot.global0;
  },
};
