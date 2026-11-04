import { useEffect, useRef, useState } from "react";
import { eventPath } from "./api";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

function label(text, x, y, z, width = 3.2, fontSize = 48) {
  const canvas = document.createElement("canvas");
  canvas.width = 768;
  canvas.height = 100;
  const context = canvas.getContext("2d");
  context.font = `500 ${fontSize}px Arial`;
  context.fillStyle = "#282052";
  context.textAlign = "center";
  context.fillText(text, 384, 62, 740);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false }),
  );
  sprite.renderOrder = 1000;
  sprite.scale.set(width, (width * 100) / 768, 1);
  sprite.position.set(x, y, z);
  return sprite;
}

function clear(group) {
  const geometries = new Set(),
    materials = new Set();
  group.traverse((o) => {
    if (o.geometry) geometries.add(o.geometry);
    if (o.material)
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) =>
        materials.add(m),
      );
  });
  geometries.forEach((g) => g.dispose());
  materials.forEach((m) => {
    m.map?.dispose();
    m.dispose();
  });
  group.clear();
}
export default function Scene({
  nodes,
  selected,
  event,
  view,
  spread,
  rotate,
  onSelect,
  resetKey,
  logScale,
  events = [],
  components = [],
  cameraMode = "orbit",
}) {
  const mount = useRef(),
    runtime = useRef(),
    selection = useRef(onSelect);
  const [failed, setFailed] = useState("");
  selection.current = onSelect;
  useEffect(() => {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailed("3D is unavailable. Use Layer Stack and the tensor inspector.");
      return;
    }
    const element = mount.current;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xfbf7f0, 0);
    element.appendChild(renderer.domElement);
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive model. Drag to orbit, right-drag to pan, scroll to zoom. Components can also be selected from the adjacent list.",
    );
    const scene = new THREE.Scene(),
      camera = new THREE.PerspectiveCamera(30, 1, 0.1, 300);
    camera.position.set(6, 7, 19);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.minDistance = 4;
    controls.maxDistance = 65;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xc2b5d1, 1.35));
    const light = new THREE.DirectionalLight(0xffffff, 1.4);
    light.position.set(8, 16, 10);
    scene.add(light);
    const fill = new THREE.DirectionalLight(0xe2d7ff, 0.75);
    fill.position.set(-10, 3, 6);
    scene.add(fill);
    const content = new THREE.Group();
    scene.add(content);
    const resize = new ResizeObserver(() => {
      const { width, height } = element.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.fov = width < 500 ? 48 : 30;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);
    let frame, down;
    const ray = new THREE.Raycaster();
    renderer.domElement.addEventListener("pointerdown", (e) => {
      down = [e.clientX, e.clientY];
    });
    renderer.domElement.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5)
        return;
      const r = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2(
          ((e.clientX - r.left) / r.width) * 2 - 1,
          (-(e.clientY - r.top) / r.height) * 2 + 1,
        ),
        camera,
      );
      const hit = ray
        .intersectObjects(content.children, true)
        .find((h) => h.object.userData.path !== undefined);
      if (hit) selection.current(hit.object.userData.path);
    });
    renderer.domElement.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      setFailed("Graphics context lost. Reload or use Layer Stack.");
    });
    function animate() {
      frame = requestAnimationFrame(animate);
      if (!document.hidden) {
        controls.update();
        renderer.render(scene, camera);
      }
    }
    animate();
    runtime.current = { camera, controls, content };
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      clear(content);
      renderer.dispose();
      renderer.domElement.remove();
      runtime.current = null;
    };
  }, []);
  useEffect(() => {
    if (runtime.current) {
      runtime.current.controls.autoRotate =
        rotate && !matchMedia("(prefers-reduced-motion: reduce)").matches;
      runtime.current.controls.autoRotateSpeed = 0.6;
    }
  }, [rotate]);
  useEffect(() => {
    if (runtime.current)
      runtime.current.controls.mouseButtons.LEFT =
        cameraMode === "pan"
          ? THREE.MOUSE.PAN
          : cameraMode === "zoom"
            ? THREE.MOUSE.DOLLY
            : THREE.MOUSE.ROTATE;
  }, [cameraMode]);
  useEffect(() => {
    const r = runtime.current;
    if (r) {
      r.camera.position.set(6, 7, 19);
      r.controls.target.set(0, 0, 0);
      r.controls.update();
    }
  }, [resetKey]);
  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    clear(rt.content);
    if (!nodes.length && !event) return;
    const geometry = new RoundedBoxGeometry(4.6, 0.12, 3.3, 2, 0.06),
      count = nodes.length,
      spacing = Math.min(0.5, 10 / Math.max(1, count)) * spread;
    const activeIndex = selected
      ? nodes.findIndex(
          (n) => n.path === selected || selected.startsWith(n.path + "."),
        )
      : -1;
    nodes.forEach((node, i) => {
      const active = i === activeIndex;
      const plate = new THREE.Mesh(
        geometry,
        new THREE.MeshPhysicalMaterial({
          color: active ? 0xc2a9f0 : 0xfffaf5,
          roughness: 0.32,
          metalness: 0,
          transparent: true,
          opacity: 1,
          depthWrite: true,
        }),
      );
      plate.position.set(-4.5, ((count - 1) / 2 - i) * spacing, 0);
      plate.rotation.y = Math.PI / 4;
      plate.userData.path = node.path;
      rt.content.add(plate);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({
          color: active ? 0x6141c4 : 0xd8ccca,
          transparent: true,
          opacity: 0.65,
        }),
      );
      edges.position.copy(plate.position);
      edges.rotation.copy(plate.rotation);
      rt.content.add(edges);
      // The first retained event for each component supplies the plate samples.
      const recorded = events.find((e) => eventPath(e) === node.path);
      const retained = (recorded?.values || []).slice(0, 64);
      if (retained.length) {
        const dots = new THREE.InstancedMesh(
          new THREE.SphereGeometry(0.047, 6, 4),
          new THREE.MeshBasicMaterial(),
          retained.length,
        );
        const transform = new THREE.Object3D();
        retained.forEach((v, j) => {
          const dx = ((j % 8) - 3.5) * 0.48;
          const dz = (Math.floor(j / 8) - 3.5) * 0.35;
          transform.position.set(
            -4.5 + (dx + dz) / Math.SQRT2,
            plate.position.y + 0.095,
            (dz - dx) / Math.SQRT2,
          );
          transform.updateMatrix();
          dots.setMatrixAt(j, transform.matrix);
          dots.setColorAt(
            j,
            new THREE.Color(
              v === null ? 0xb7b3b7 : v < 0 ? 0xf08b83 : 0x69bdaa,
            ),
          );
        });
        rt.content.add(dots);
      }
    });
    const values = (event?.values || []).flat().slice(0, 2048),
      finite = values.filter((v) => v !== null && Number.isFinite(v));
    const scale = Math.max(...finite.map(Math.abs), 0.00001);
    const normalize = (v) =>
      logScale
        ? (Math.sign(v) * Math.log1p(Math.abs(v))) / Math.log1p(scale)
        : v / scale;
    const expanded = view !== "volume" && components.length > 0;
    if (expanded) {
      // The translucent enclosure is a visual grouping of containment, not a
      // claim about execution order or the model's physical structure.
      const enclosure = new THREE.Mesh(
        new RoundedBoxGeometry(10.0, 7.0, 1.0, 5, 0.35),
        new THREE.MeshPhysicalMaterial({
          color: 0xc9b0ed,
          transparent: true,
          opacity: 0.3,
          roughness: 0.28,
          clearcoat: 1,
          depthWrite: false,
        }),
      );
      enclosure.position.set(3.7, 1, -0.25);
      rt.content.add(enclosure);
      const enclosureOutline = new THREE.LineSegments(
        new THREE.EdgesGeometry(enclosure.geometry),
        new THREE.LineBasicMaterial({
          color: 0xb6a1ce,
          transparent: true,
          opacity: 0.65,
        }),
      );
      enclosureOutline.position.copy(enclosure.position);
      rt.content.add(enclosureOutline);
      if (activeIndex >= 0) {
        const name = nodes[activeIndex].path.split(".").slice(-2).join(".");
        const badge = new THREE.Mesh(
          new RoundedBoxGeometry(4.1, 0.58, 0.2, 4, 0.14),
          new THREE.MeshPhysicalMaterial({ color: 0xe9dcf7, roughness: 0.32 }),
        );
        badge.position.set(3.7, 4.75, 0.55);
        rt.content.add(badge);
        rt.content.add(label(name, 3.7, 4.76, 0.8, 3.3, 54));
      }
      components.slice(0, 2).forEach((component, index) => {
        const x = 3.7;
        const y = 2.55 - index * 3.1;
        const card = new THREE.Mesh(
          new RoundedBoxGeometry(9.1, 2.7, 0.4, 6, 0.22),
          new THREE.MeshPhysicalMaterial({
            color: component.path === selected ? 0xc4a8f1 : 0xf6f0fb,
            roughness: 0.35,
            metalness: 0,
            clearcoat: 1,
            transparent: true,
            opacity: 0.9,
          }),
        );
        card.position.set(x, y, 0.6);
        card.userData.path = component.path;
        rt.content.add(card);
        const cardOutline = new THREE.LineSegments(
          new THREE.EdgesGeometry(card.geometry),
          new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.85,
          }),
        );
        cardOutline.position.copy(card.position);
        rt.content.add(cardOutline);
        const title = component.path.split(".").at(-1).replaceAll("_", " ");
        rt.content.add(
          label(
            `${title} · ${component.type?.split(".").at(-1) || "Component"}`,
            x,
            y + 0.9,
            1.1,
            6.2,
          ),
        );
        (component.children || []).slice(0, 4).forEach((child, j, array) => {
          const childX = x + (j - (array.length - 1) / 2) * 2.05;
          const blockGeometry = new THREE.BoxGeometry(1.18, 0.75, 0.65);
          const block = new THREE.Mesh(
            blockGeometry,
            new THREE.MeshPhysicalMaterial({
              color: child.path === selected ? 0x7657cc : 0xa995f3,
              roughness: 0.35,
              clearcoat: 1,
            }),
          );
          block.position.set(childX, y - 0.03, 1.3);
          block.userData.path = child.path;
          rt.content.add(block);
          // A regular surface grid conveys the dimensionality of the selectable
          // module tile. It is decorative, not a representation of its weights.
          const lines = [];
          for (let k = 0; k <= 4; k++) {
            const px = -0.59 + (1.18 * k) / 4;
            const py = -0.375 + (0.75 * k) / 4;
            lines.push(px, -0.375, 0.326, px, 0.375, 0.326);
            lines.push(-0.59, py, 0.326, 0.59, py, 0.326);
          }
          const gridGeometry = new THREE.BufferGeometry();
          gridGeometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(lines, 3),
          );
          const grid = new THREE.LineSegments(
            gridGeometry,
            new THREE.LineBasicMaterial({
              color: 0xffffff,
              transparent: true,
              opacity: 0.65,
            }),
          );
          grid.position.copy(block.position);
          rt.content.add(grid);
          rt.content.add(
            label(child.path.split(".").at(-1), childX, y - 0.65, 1.8, 2.1, 80),
          );
        });
      });
    }
    if (values.length && !expanded) {
      const volume = view === "volume",
        columns = volume
          ? Math.ceil(Math.cbrt(values.length))
          : Math.ceil(Math.sqrt(values.length)),
        unit = Math.min(0.38, 7.6 / columns),
        rows = volume ? columns : Math.ceil(values.length / columns);
      const geo = volume
        ? new THREE.BoxGeometry(unit * 0.72, unit * 0.72, unit * 0.72)
        : new THREE.BoxGeometry(unit * 0.82, unit * 0.82, unit * 0.56);
      const mesh = new THREE.InstancedMesh(
          geo,
          new THREE.MeshBasicMaterial(),
          values.length,
        ),
        dummy = new THREE.Object3D(),
        color = new THREE.Color();
      values.forEach((v, i) => {
        const x = i % columns,
          y = Math.floor(i / columns) % rows,
          z = volume ? Math.floor(i / (columns * rows)) : 0;
        dummy.position.set(
          3 + (x - (columns - 1) / 2) * unit,
          volume
            ? (z - (columns - 1) / 2) * unit
            : expanded
              ? -2.65 + ((rows - 1) / 2 - y) * Math.min(unit, 3.3 / rows)
              : ((rows - 1) / 2 - y) * unit,
          volume
            ? (y - (rows - 1) / 2) * unit
            : 0.8 + (v === null ? 0 : normalize(v)) * 0.8,
        );
        dummy.updateMatrix();
        if (expanded) {
          dummy.scale.y = Math.min(1, 3.3 / (rows * unit));
          dummy.updateMatrix();
        }
        mesh.setMatrixAt(i, dummy.matrix);
        color.set(v === null ? 0xb7b3b7 : v < 0 ? 0xef746b : 0x239f8b);
        if (v !== null)
          color.lerp(
            new THREE.Color(0xf1ebe4),
            0.78 * (1 - Math.min(1, Math.abs(normalize(v)))),
          );
        mesh.setColorAt(i, color);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      rt.content.add(mesh);
      const base = new THREE.Mesh(
        volume
          ? new THREE.BoxGeometry(7.2, 0.1, 7.2)
          : new RoundedBoxGeometry(
              8.2,
              expanded ? 3.9 : Math.max(3, rows * unit + 0.6),
              0.18,
              3,
              0.08,
            ),
        new THREE.MeshPhysicalMaterial({
          color: 0xded3ef,
          transparent: true,
          opacity: 0.28,
          roughness: 0.4,
          depthWrite: false,
        }),
      );
      base.position.set(
        3,
        volume ? -2.4 : expanded ? -2.65 : 0,
        volume ? 0 : -0.2,
      );
      rt.content.add(base);
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(base.geometry),
        new THREE.LineBasicMaterial({ color: 0xa595cb }),
      );
      outline.position.copy(base.position);
      rt.content.add(outline);
    } else if (!expanded) {
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry(6.5, 0.25, 5.8),
        new THREE.MeshPhysicalMaterial({
          color: 0xb2a3d9,
          roughness: 0.35,
          transparent: true,
          opacity: 0.45,
        }),
      );
      slab.position.set(3, 0, 0);
      rt.content.add(slab);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(slab.geometry),
        new THREE.LineBasicMaterial({ color: 0x8875b8 }),
      );
      edges.position.copy(slab.position);
      rt.content.add(edges);
    }
  }, [nodes, selected, event, view, spread, logScale, events, components]);
  return (
    <div className="scene-mount" ref={mount}>
      {failed && (
        <p className="scene-error" role="status">
          {failed}
        </p>
      )}
    </div>
  );
}
