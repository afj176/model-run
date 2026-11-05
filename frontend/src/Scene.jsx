import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { eventPath } from "./api";
import { activationStats, railLabel } from "./insights";

/** The Run View stage: a model map (one slab per block or step, tinted by its
 * captured activity) linked to a floating field of the selected event's values.
 * Same dark stage, mint bars and pink selection as the exploration table. */

const HOME = new THREE.Vector3(-4.2, 10.5, 22);
const TARGET = new THREE.Vector3(0.6, 1.4, 0);
const STACK_X = -5.6;
const FIELD_X = 4.2;
const FIELD_Y = 1.7;
const FIELD_W = 9;

const C = {
  floor: 0x1b1536,
  grid: 0x8b7fd1,
  slab: 0x2a2357,
  slabEdge: 0x5c4f9c,
  slabHover: 0xc9bdf4,
  low: 0xf2efe4,
  mid: 0x7fd9c2,
  positive: 0x0f9f86,
  negative: 0xf16d83,
  pink: 0xff3d8f,
  outline: 0xffd3e6,
  tile: 0x4a3c8f,
  tileEdge: 0x9b8cf0,
  null: 0x5a5478,
};

function makeLabel(text, options = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  const size = options.size || 34;
  context.font = `${options.weight || 500} ${size}px "Segoe UI", system-ui, sans-serif`;
  context.textAlign = options.align || "center";
  context.textBaseline = "middle";
  const x =
    options.align === "left" ? 8 : options.align === "right" ? 504 : 256;
  const content = String(text);
  if (options.highlight) {
    const w = Math.min(500, context.measureText(content).width + 36);
    const left =
      options.align === "left"
        ? 0
        : options.align === "right"
          ? 512 - w
          : 256 - w / 2;
    context.fillStyle = "#ff3d8f";
    context.beginPath();
    context.roundRect(left, 20, w, 56, 16);
    context.fill();
    context.fillStyle = "#fff6fa";
  } else context.fillStyle = options.color || "#ebe6ff";
  context.fillText(content, x, 48, 500);
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 4;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map, depthTest: false, transparent: true }),
  );
  const scale = options.scale || 1;
  sprite.scale.set(3.2 * scale, 0.6 * scale, 1);
  sprite.renderOrder = 100;
  return sprite;
}

function edges(geometry, color, opacity = 0.9) {
  return new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}

function disposeGroup(group) {
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
  valueIndex = null,
  onPickValue,
  callout = null,
}) {
  const mount = useRef(),
    calloutRef = useRef(null),
    runtime = useRef(),
    selection = useRef(onSelect),
    pickValue = useRef(onPickValue);
  const [failed, setFailed] = useState("");
  selection.current = onSelect;
  pickValue.current = onPickValue;

  useEffect(() => {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailed("3D is unavailable. Use Layer Stack and the tensor inspector.");
      return;
    }
    const element = mount.current;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0e0a21, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.domElement.style.cssText =
      "display:block;width:100%;height:100%;touch-action:none";
    element.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive model map. Click a slab to select a block or step, click a bar to select a value. Drag to orbit, right-drag to pan, scroll to zoom.",
    );
    const scene = new THREE.Scene(),
      camera = new THREE.PerspectiveCamera(27, 1, 0.1, 200);
    camera.position.copy(HOME);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.minDistance = 5;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.copy(TARGET);
    scene.add(new THREE.HemisphereLight(0xe9e4ff, 0x2a1f4d, 1.5));
    const key = new THREE.DirectionalLight(0xfff6ea, 2.3);
    key.position.set(-7, 14, 10);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x6ef0d2, 0.9);
    fill.position.set(9, 5, -8);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xc8a8ff, 0.7);
    rim.position.set(0, 6, -14);
    scene.add(rim);
    const content = new THREE.Group();
    scene.add(content);

    // Picking state is replaced on every rebuild.
    const pick = { clickable: [], bars: null, barIndex: [], anchor: null };
    let hovered = null;
    const ray = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const toPointer = (e) => {
      const r = renderer.domElement.getBoundingClientRect();
      if (!r.width || !r.height) return false;
      pointer.set(
        ((e.clientX - r.left) / r.width) * 2 - 1,
        1 - ((e.clientY - r.top) / r.height) * 2,
      );
      ray.setFromCamera(pointer, camera);
      return true;
    };
    let down = null;
    const pointerDown = (e) => {
      down = [e.clientX, e.clientY];
    };
    const pointerUp = (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5)
        return;
      if (!toPointer(e)) return;
      const slab = ray.intersectObjects(pick.clickable, false)[0];
      const bar = pick.bars ? ray.intersectObject(pick.bars)[0] : null;
      if (bar && (!slab || bar.distance < slab.distance)) {
        const index = pick.barIndex[bar.instanceId];
        if (index !== undefined) pickValue.current?.(index);
        return;
      }
      if (slab) selection.current?.(slab.object.userData.path);
    };
    const pointerMove = (e) => {
      if (!toPointer(e)) return;
      const hit = ray.intersectObjects(pick.clickable, false)[0];
      const next = hit ? hit.object : null;
      if (next !== hovered) {
        if (hovered?.userData.edge)
          hovered.userData.edge.material.color.set(hovered.userData.edgeColor);
        hovered = next;
        if (hovered?.userData.edge)
          hovered.userData.edge.material.color.set(C.slabHover);
      }
      const overBar = !hit && pick.bars && ray.intersectObject(pick.bars)[0];
      renderer.domElement.style.cursor = hit || overBar ? "pointer" : "";
    };
    const contextLost = (e) => {
      e.preventDefault();
      setFailed("Graphics context lost. Reload or use Layer Stack.");
    };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    renderer.domElement.addEventListener("pointermove", pointerMove);
    renderer.domElement.addEventListener("webglcontextlost", contextLost);
    let size = { width: 0, height: 0 };
    const resize = new ResizeObserver(() => {
      const { width, height } = element.getBoundingClientRect();
      if (!width || !height) return;
      size = { width, height };
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.fov = width < 500 ? 46 : 27;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);
    let frame;
    const projected = new THREE.Vector3();
    let lastKey = "";
    const animate = () => {
      frame = requestAnimationFrame(animate);
      if (document.hidden) return;
      controls.update();
      renderer.render(scene, camera);
      const el = calloutRef.current;
      if (!el) return;
      if (pick.anchor && size.width) {
        projected.copy(pick.anchor).project(camera);
        const x = ((projected.x + 1) / 2) * size.width;
        const y = ((1 - projected.y) / 2) * size.height;
        const next = `${Math.round(x)}:${Math.round(y)}`;
        if (next !== lastKey) {
          lastKey = next;
          el.style.transform = `translate(${x}px, ${y}px)`;
          el.style.opacity = projected.z < 1 ? "1" : "0";
        }
      } else if (lastKey !== "none") {
        lastKey = "none";
        el.style.opacity = "0";
      }
    };
    animate();
    runtime.current = { camera, controls, content, pick };
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("pointermove", pointerMove);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      controls.dispose();
      disposeGroup(content);
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
      r.camera.position.copy(HOME);
      r.controls.target.copy(TARGET);
      r.controls.update();
    }
  }, [resetKey]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeGroup(rt.content);
    rt.pick.clickable = [];
    rt.pick.bars = null;
    rt.pick.barIndex = [];
    rt.pick.anchor = null;
    rt.field = null;
    rt.overlay = null;
    if (!nodes.length && !event) return;
    const add = (o) => rt.content.add(o);
    const color = new THREE.Color();
    const low = new THREE.Color(C.low),
      mid = new THREE.Color(C.mid),
      positive = new THREE.Color(C.positive),
      negative = new THREE.Color(C.negative),
      pink = new THREE.Color(C.pink);
    const volume = view === "volume";

    // Floor and grid
    const floorW = 26,
      floorD = 16;
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry(floorW, 0.18, floorD),
      new THREE.MeshStandardMaterial({
        color: C.floor,
        roughness: 0.95,
        metalness: 0.05,
      }),
    );
    floor.position.set(-0.5, -0.1, 0);
    add(floor);
    const gridVertices = [];
    for (let x = -floorW / 2; x <= floorW / 2; x += 1)
      gridVertices.push(
        x - 0.5,
        0.002,
        -floorD / 2,
        x - 0.5,
        0.002,
        floorD / 2,
      );
    for (let z = -floorD / 2; z <= floorD / 2; z += 1)
      gridVertices.push(
        -floorW / 2 - 0.5,
        0.002,
        z,
        floorW / 2 - 0.5,
        0.002,
        z,
      );
    const grid = new THREE.BufferGeometry();
    grid.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(gridVertices, 3),
    );
    add(
      new THREE.LineSegments(
        grid,
        new THREE.LineBasicMaterial({
          color: C.grid,
          transparent: true,
          opacity: 0.16,
        }),
      ),
    );

    // Values of the selected event
    const raw = (event?.values || []).flat().slice(0, 2048);
    const finite = raw.filter((v) => v !== null && Number.isFinite(v));
    const scale = Math.max(...finite.map(Math.abs), 0.00001);
    const normalize = (v) =>
      logScale
        ? (Math.sign(v) * Math.log1p(Math.abs(v))) / Math.log1p(scale)
        : v / scale;
    const tint = (v, out) => {
      if (v === null || !Number.isFinite(v)) return out.set(C.null);
      const n = Math.min(1, Math.abs(normalize(v)));
      if (v < 0) return out.copy(low).lerp(negative, Math.max(0.15, n));
      if (n < 0.5) return out.copy(low).lerp(mid, n * 2);
      return out.copy(mid).lerp(positive, (n - 0.5) * 2);
    };

    // Model map: one slab per node, tinted by that node's captured RMS.
    const activeIndex = selected
      ? nodes.findIndex(
          (n) => n.path === selected || selected.startsWith(n.path + "."),
        )
      : -1;
    const firstEvent = new Map();
    for (const e of events) {
      const key = eventPath(e);
      if (!firstEvent.has(key)) firstEvent.set(key, e);
    }
    const rmsByNode = nodes.map((n) => {
      const s = activationStats(firstEvent.get(n.path)?.values);
      return s ? s.rms : null;
    });
    const maxRms = Math.max(1e-9, ...rmsByNode.filter((v) => v !== null));
    let selectedSlab = null;
    if (!volume && nodes.length) {
      const count = nodes.length;
      const spacing = Math.min(0.46, 10.5 / Math.max(1, count)) * spread;
      const slabGeometry = new RoundedBoxGeometry(4.4, 0.12, 3.2, 2, 0.05);
      nodes.forEach((node, i) => {
        const active = i === activeIndex;
        const rms = rmsByNode[i];
        const captured = rms !== null;
        const intensity = captured ? 0.25 + 0.75 * Math.sqrt(rms / maxRms) : 0;
        color.set(C.slab);
        if (captured) color.lerp(positive, intensity * 0.8);
        if (active) color.lerp(pink, 0.5);
        const slab = new THREE.Mesh(
          slabGeometry,
          new THREE.MeshStandardMaterial({
            color: color.clone(),
            roughness: 0.55,
            metalness: 0.1,
            emissive: active ? C.pink : captured ? C.positive : 0x000000,
            emissiveIntensity: active ? 0.35 : captured ? intensity * 0.18 : 0,
          }),
        );
        const y = ((count - 1) / 2 - i) * spacing + 1.4;
        const z = active ? 1.1 : 0;
        slab.position.set(STACK_X, y, z);
        slab.userData.path = node.path;
        const edgeColor = active ? C.pink : captured ? C.mid : C.slabEdge;
        const edge = edges(slabGeometry, edgeColor, active ? 1 : 0.55);
        edge.position.copy(slab.position);
        slab.userData.edge = edge;
        slab.userData.edgeColor = edgeColor;
        add(slab);
        add(edge);
        rt.pick.clickable.push(slab);
        if (active) selectedSlab = slab;
        // A strip of the node's first retained values sits on the slab.
        const strip = (firstEvent.get(node.path)?.values || []).slice(0, 24);
        if (strip.length) {
          const stripScale = Math.max(
            ...strip.filter((v) => v !== null).map(Math.abs),
            1e-9,
          );
          const bars = new THREE.InstancedMesh(
            new THREE.BoxGeometry(0.13, 1, 0.5),
            new THREE.MeshStandardMaterial({ roughness: 0.5 }),
            strip.length,
          );
          const t = new THREE.Object3D();
          strip.forEach((v, j) => {
            const h =
              v === null ? 0.03 : 0.04 + (Math.abs(v) / stripScale) * 0.42;
            t.position.set(
              STACK_X - 1.9 + (j / (strip.length - 1 || 1)) * 3.8,
              y + 0.06 + h / 2,
              z - 0.9,
            );
            t.scale.set(1, h, 1);
            t.updateMatrix();
            bars.setMatrixAt(j, t.matrix);
            bars.setColorAt(
              j,
              v === null
                ? color.set(C.null)
                : v < 0
                  ? color.copy(negative)
                  : color.copy(mid).lerp(positive, Math.abs(v) / stripScale),
            );
          });
          bars.instanceMatrix.needsUpdate = true;
          add(bars);
        }
        if (active) {
          const text = active
            ? `${railLabel(node.path, i)} · ${node.path.split(".").slice(-2).join(".")}`
            : railLabel(node.path, i);
          const sprite = makeLabel(text, {
            highlight: active,
            align: "right",
            size: active ? 34 : 28,
            color: captured ? "#cfeee6" : "#9d93c9",
            scale: active ? 1.25 : 0.85,
          });
          sprite.position.set(
            STACK_X - 2.4 - (active ? 1.6 : 1.1),
            y + 0.05,
            z + 0.2,
          );
          add(sprite);
        }
      });
    }

    // Children of the selected block as tiles behind the field.
    const tiles = volume ? [] : components.slice(0, 8);
    if (tiles.length) {
      const pitch = Math.min(1.75, 9 / tiles.length);
      tiles.forEach((component, j) => {
        const captured = firstEvent.has(component.path);
        const geometry = new RoundedBoxGeometry(
          Math.min(1.5, pitch * 0.82),
          0.9,
          0.5,
          2,
          0.08,
        );
        const picked = component.path === selected;
        const tile = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: picked ? C.pink : captured ? C.positive : C.tile,
            roughness: 0.5,
            emissive: picked ? C.pink : captured ? C.positive : 0x000000,
            emissiveIntensity: picked ? 0.3 : captured ? 0.15 : 0,
          }),
        );
        const x = FIELD_X + (j - (tiles.length - 1) / 2) * pitch;
        tile.position.set(x, FIELD_Y + 2.3, -5.8);
        tile.userData.path = component.path;
        const edgeColor = picked ? C.outline : C.tileEdge;
        const edge = edges(geometry, edgeColor, 0.9);
        edge.position.copy(tile.position);
        tile.userData.edge = edge;
        tile.userData.edgeColor = edgeColor;
        add(tile);
        add(edge);
        rt.pick.clickable.push(tile);
        const sprite = makeLabel(component.path.split(".").at(-1), {
          size: 26,
          color: picked ? "#ffd3e6" : captured ? "#cfeee6" : "#b9b0dd",
          scale: 0.75,
        });
        sprite.position.set(x, FIELD_Y + 1.62, -5.8);
        add(sprite);
      });
      const title = makeLabel(
        `inside ${selected.split(".").slice(-2).join(".")} · ${tiles.length} components`,
        { size: 26, color: "#b9b0dd", scale: 1.1 },
      );
      title.position.set(FIELD_X, FIELD_Y + 3.1, -5.8);
      add(title);
    }

    if (!raw.length) {
      if (selectedSlab) {
        const note = makeLabel("no values captured for this block", {
          size: 26,
          color: "#9d93c9",
          scale: 1.2,
        });
        note.position.set(FIELD_X, FIELD_Y + 0.6, 0);
        add(note);
      }
      return;
    }

    // Value field: the event's retained prefix folded into a grid of bars.
    const n = raw.length;
    const cols = volume
      ? Math.ceil(Math.cbrt(n))
      : (event?.grid_cols ?? Math.ceil(Math.sqrt(n)));
    const rows = volume ? cols : Math.ceil(n / cols);
    const cell = FIELD_W / Math.max(cols, 1);
    const depth = rows * cell;
    const fieldY = volume ? 0.6 : FIELD_Y;
    const x0 = FIELD_X - FIELD_W / 2;
    const z0 = -depth / 2;
    const maxHeight = volume ? cell * 0.8 : 2.6;
    const bars = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.05 }),
      n,
    );
    const t = new THREE.Object3D();
    const inset = cols > 40 ? 0.74 : 0.66;
    raw.forEach((v, i) => {
      const cx = i % cols;
      const cy = Math.floor(i / cols) % rows;
      const cz = volume ? Math.floor(i / (cols * rows)) : 0;
      const nv = v === null || !Number.isFinite(v) ? 0 : normalize(v);
      if (volume) {
        const s = cell * (0.25 + 0.6 * Math.abs(nv));
        t.position.set(
          x0 + (cx + 0.5) * cell,
          fieldY + (cz + 0.5) * cell,
          z0 + (cy + 0.5) * cell,
        );
        t.scale.set(s, s, s);
      } else {
        const h = Math.max(0.04, Math.abs(nv) * maxHeight);
        const direction = v !== null && v < 0 ? -1 : 1;
        t.position.set(
          x0 + (cx + 0.5) * cell,
          fieldY + (direction * h) / 2,
          z0 + depth - (cy + 0.5) * cell,
        );
        t.scale.set(cell * inset, h, cell * inset);
      }
      t.updateMatrix();
      bars.setMatrixAt(i, t.matrix);
      tint(v, color);
      bars.setColorAt(i, color);
      rt.pick.barIndex[i] = i;
    });
    bars.instanceMatrix.needsUpdate = true;
    if (bars.instanceColor) bars.instanceColor.needsUpdate = true;
    add(bars);
    rt.pick.bars = bars;
    rt.field = {
      bars,
      raw,
      cols,
      rows,
      cell,
      depth,
      fieldY,
      x0,
      z0,
      maxHeight,
      volume,
      normalize,
      tint,
      n,
    };
    rt.overlay = new THREE.Group();
    add(rt.overlay);

    if (!volume) {
      // Floating platform with its own grid, linked to the selected slab.
      const platform = new THREE.Mesh(
        new RoundedBoxGeometry(FIELD_W + 0.8, 0.16, depth + 0.8, 2, 0.06),
        new THREE.MeshStandardMaterial({
          color: 0x241d4d,
          roughness: 0.9,
          transparent: true,
          opacity: 0.92,
        }),
      );
      platform.position.set(FIELD_X, fieldY - 0.09, 0);
      add(platform);
      const platformEdge = edges(platform.geometry, C.grid, 0.5);
      platformEdge.position.copy(platform.position);
      add(platformEdge);
      const lines = [];
      for (let c = 0; c <= cols; c++)
        lines.push(
          x0 + c * cell,
          fieldY + 0.002,
          z0,
          x0 + c * cell,
          fieldY + 0.002,
          z0 + depth,
        );
      for (let r = 0; r <= rows; r++)
        lines.push(
          x0,
          fieldY + 0.002,
          z0 + r * cell,
          x0 + FIELD_W,
          fieldY + 0.002,
          z0 + r * cell,
        );
      const fieldGrid = new THREE.BufferGeometry();
      fieldGrid.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(lines, 3),
      );
      add(
        new THREE.LineSegments(
          fieldGrid,
          new THREE.LineBasicMaterial({
            color: C.grid,
            transparent: true,
            opacity: 0.22,
          }),
        ),
      );
      const caption = makeLabel(
        `${event.label ?? eventPath(event)} · ${n.toLocaleString()} values · ${cols}×${rows}`,
        { size: 26, color: "#b9b0dd", scale: 1.2 },
      );
      caption.position.set(FIELD_X, fieldY - 0.45, depth / 2 + 0.9);
      add(caption);
      if (selectedSlab) {
        const from = selectedSlab.position
          .clone()
          .add(new THREE.Vector3(2.2, 0.06, 0));
        const to = new THREE.Vector3(x0 - 0.4, fieldY, 0);
        const linkGeometry = new THREE.BufferGeometry().setFromPoints([
          from,
          to,
        ]);
        add(
          new THREE.Line(
            linkGeometry,
            new THREE.LineBasicMaterial({
              color: C.pink,
              transparent: true,
              opacity: 0.7,
            }),
          ),
        );
        const ends = [from, to].map((p) => {
          const dot = new THREE.Mesh(
            new THREE.SphereGeometry(0.09, 10, 8),
            new THREE.MeshBasicMaterial({ color: C.pink }),
          );
          dot.position.copy(p);
          return dot;
        });
        ends.forEach(add);
      }
    }
  }, [nodes, selected, event, view, spread, logScale, events, components]);

  // Selection overlay: recolour bars and rebuild only the bands, outline and callout anchor.
  useEffect(() => {
    const rt = runtime.current;
    if (!rt?.field || !rt.overlay) return;
    const {
      bars,
      raw,
      cols,
      rows,
      cell,
      depth,
      fieldY,
      x0,
      z0,
      maxHeight,
      volume,
      normalize,
      tint,
      n,
    } = rt.field;
    disposeGroup(rt.overlay);
    rt.pick.anchor = null;
    const color = new THREE.Color();
    const pink = new THREE.Color(C.pink);
    raw.forEach((v, i) => {
      tint(v, color);
      if (valueIndex !== null) {
        const cx = i % cols;
        const cy = Math.floor(i / cols) % rows;
        if (i === valueIndex) color.copy(pink);
        else if (
          !volume &&
          (cx === valueIndex % cols || cy === Math.floor(valueIndex / cols))
        )
          color.lerp(pink, 0.3);
      }
      bars.setColorAt(i, color);
    });
    if (bars.instanceColor) bars.instanceColor.needsUpdate = true;
    const add = (o) => rt.overlay.add(o);
    if (valueIndex !== null && valueIndex < n) {
      const v = raw[valueIndex];
      const cx = valueIndex % cols;
      const cy = Math.floor(valueIndex / cols) % rows;
      const cz = volume ? Math.floor(valueIndex / (cols * rows)) : 0;
      const nv = v === null || !Number.isFinite(v) ? 0 : normalize(v);
      let position, outlineSize;
      if (volume) {
        position = new THREE.Vector3(
          x0 + (cx + 0.5) * cell,
          fieldY + (cz + 0.5) * cell,
          z0 + (cy + 0.5) * cell,
        );
        outlineSize = [cell * 1.05, cell * 1.05, cell * 1.05];
      } else {
        const h = Math.max(0.04, Math.abs(nv) * maxHeight);
        const direction = v !== null && v < 0 ? -1 : 1;
        position = new THREE.Vector3(
          x0 + (cx + 0.5) * cell,
          fieldY + (direction * h) / 2,
          z0 + depth - (cy + 0.5) * cell,
        );
        outlineSize = [cell * 1.12, h + 0.06, cell * 1.12];
        const bandMaterial = new THREE.MeshBasicMaterial({
          color: C.pink,
          transparent: true,
          opacity: 0.16,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        const rowBand = new THREE.Mesh(
          new THREE.PlaneGeometry(FIELD_W + 0.6, cell),
          bandMaterial,
        );
        rowBand.rotation.x = -Math.PI / 2;
        rowBand.position.set(FIELD_X, fieldY + 0.012, position.z);
        add(rowBand);
        const colBand = new THREE.Mesh(
          new THREE.PlaneGeometry(cell, depth + 0.6),
          bandMaterial,
        );
        colBand.rotation.x = -Math.PI / 2;
        colBand.position.set(position.x, fieldY + 0.012, 0);
        add(colBand);
      }
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(...outlineSize)),
        new THREE.LineBasicMaterial({ color: C.outline, depthTest: false }),
      );
      outline.position.copy(position);
      outline.renderOrder = 90;
      add(outline);
      rt.pick.anchor = position
        .clone()
        .add(new THREE.Vector3(0, outlineSize[1] / 2 + 0.05, 0));
    }
  }, [
    valueIndex,
    nodes,
    selected,
    event,
    view,
    spread,
    logScale,
    events,
    components,
  ]);

  return (
    <div className="scene-mount" ref={mount}>
      {failed && (
        <p className="scene-error" role="status">
          {failed}
        </p>
      )}
      {callout && !failed && (
        <div ref={calloutRef} className="matrix-callout-anchor">
          <div className="matrix-callout">{callout}</div>
        </div>
      )}
    </div>
  );
}
