import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { guardWheel } from "./wheelGuard";

// All geometry is derived from the provided matrix; nulls are missing, not zero.
// The container should provide a definite width and height.
const HOME = new THREE.Vector3(-9.2, 14.2, 23.2);
const TARGET = new THREE.Vector3(-0.9, 0.5, 0.6);

export default function MatrixScene({
  matrix,
  selected,
  onSelect,
  mode = "attention",
  resetKey = 0,
  zoom = 1,
  axisLabels = { row: "row index", col: "column index" },
  cameraMode = "orbit",
  callout = null,
}) {
  const mount = useRef(null);
  const calloutRef = useRef(null);
  const runtime = useRef(null);
  const selectRef = useRef(onSelect);
  const [failure, setFailure] = useState("");
  selectRef.current = onSelect;

  useEffect(() => {
    const host = mount.current;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailure("3D is unavailable. Use the matrix table to inspect values.");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0e0a21, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.domElement.style.cssText =
      "display:block;width:100%;height:100%;touch-action:none";
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 3D matrix. Drag to orbit, right-drag to pan, and Ctrl + scroll (or click first, then scroll) to zoom. Use the heatmap for keyboard selection.",
    );
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(27, 1, 0.1, 150);
    camera.position.copy(HOME);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.minDistance = 5;
    controls.maxDistance = 48;
    const releaseWheel = guardWheel(renderer.domElement, controls);
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.copy(TARGET);
    scene.add(new THREE.HemisphereLight(0xe9e4ff, 0x2a1f4d, 1.5));
    const key = new THREE.DirectionalLight(0xfff6ea, 2.4);
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
    let activeMesh = null;
    let coordinates = [];
    let anchor = null;
    const setPickData = (mesh, positions, anchorPoint) => {
      activeMesh = mesh;
      coordinates = positions;
      anchor = anchorPoint;
    };
    let down = null;
    const pointerDown = (event) => {
      down = [event.clientX, event.clientY];
    };
    const pointerUp = (event) => {
      if (
        !down ||
        !activeMesh ||
        Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5
      )
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const ray = new THREE.Raycaster();
      ray.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          1 - ((event.clientY - rect.top) / rect.height) * 2,
        ),
        camera,
      );
      const hit = ray.intersectObject(activeMesh)[0];
      if (hit?.instanceId !== undefined && coordinates[hit.instanceId]) {
        const [row, col] = coordinates[hit.instanceId];
        selectRef.current?.(row, col);
      }
    };
    const contextLost = (event) => {
      event.preventDefault();
      setFailure("Graphics context lost. Reload or use the matrix table.");
    };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    renderer.domElement.addEventListener("webglcontextlost", contextLost);
    let size = { width: 0, height: 0 };
    const resize = new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      if (!width || !height) return;
      size = { width, height };
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.fov = width < 500 ? 46 : 27;
      camera.updateProjectionMatrix();
    });
    resize.observe(host);
    let frame;
    const projected = new THREE.Vector3();
    let lastKey = "";
    const animate = () => {
      frame = requestAnimationFrame(animate);
      if (document.hidden) return;
      controls.update();
      renderer.render(scene, camera);
      const el = calloutRef.current;
      if (el) {
        if (anchor && size.width) {
          projected.copy(anchor).project(camera);
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
      }
    };
    runtime.current = { content, controls, camera, setPickData };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      releaseWheel();
      controls.dispose();
      disposeGroup(content);
      renderer.dispose();
      renderer.domElement.remove();
      runtime.current = null;
    };
  }, []);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    rt.camera.position.copy(HOME);
    rt.controls.target.copy(TARGET);
    rt.controls.update();
  }, [resetKey]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt || !(zoom > 0)) return;
    const offset = rt.camera.position.clone().sub(rt.controls.target);
    const base = HOME.clone().sub(TARGET).length();
    offset.setLength(
      Math.max(
        rt.controls.minDistance,
        Math.min(rt.controls.maxDistance, base / zoom),
      ),
    );
    rt.camera.position.copy(rt.controls.target).add(offset);
    rt.controls.update();
  }, [zoom]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    rt.controls.mouseButtons.LEFT =
      cameraMode === "pan"
        ? THREE.MOUSE.PAN
        : cameraMode === "zoom"
          ? THREE.MOUSE.DOLLY
          : THREE.MOUSE.ROTATE;
  }, [cameraMode]);

  useEffect(() => {
    const rt = runtime.current;
    if (!rt) return;
    disposeGroup(rt.content);
    rt.setPickData(null, [], null);
    if (!Array.isArray(matrix) || !matrix.length || !Array.isArray(matrix[0]))
      return;
    const rows = matrix.length;
    const cols = matrix[0].length;
    if (
      !cols ||
      rows > 128 ||
      cols > 128 ||
      matrix.some((row) => !Array.isArray(row) || row.length !== cols)
    ) {
      setFailure(
        "The 3D view requires a rectangular slice of at most 128 × 128 values.",
      );
      return;
    }
    const cells = [];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const value = matrix[row][col];
        if (typeof value === "number" && Number.isFinite(value))
          cells.push([row, col, value]);
      }
    }
    if (
      mode === "attention" &&
      cells.some(([, , value]) => value < 0 || value > 1)
    ) {
      setFailure(
        "Attention weights must be in the range 0–1. Check the tensor type.",
      );
      return;
    }
    setFailure("");
    const width = 12,
      depth = 10;
    const cellWidth = width / cols,
      cellDepth = depth / rows;
    const x0 = -width / 2,
      z0 = -depth / 2;
    const cellX = (col) => x0 + (col + 0.5) * cellWidth;
    const cellZ = (row) => z0 + depth - (row + 0.5) * cellDepth;
    const pink = new THREE.Color(0xff3d8f);

    // Floor slab and grid
    const floor = new THREE.Mesh(
      new THREE.BoxGeometry(width + 1.2, 0.18, depth + 1.2),
      new THREE.MeshStandardMaterial({
        color: 0x1b1536,
        roughness: 0.95,
        metalness: 0.05,
      }),
    );
    floor.position.set(0, -0.1, 0);
    rt.content.add(floor);
    const gridVertices = [];
    const addLine = (...coords) => gridVertices.push(...coords);
    for (let c = 0; c <= cols; c++) {
      const x = x0 + c * cellWidth;
      addLine(x, 0.002, z0, x, 0.002, z0 + depth);
    }
    for (let r = 0; r <= rows; r++) {
      const z = z0 + r * cellDepth;
      addLine(x0, 0.002, z, x0 + width, 0.002, z);
    }
    const grid = new THREE.BufferGeometry();
    grid.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(gridVertices, 3),
    );
    rt.content.add(
      new THREE.LineSegments(
        grid,
        new THREE.LineBasicMaterial({
          color: 0x8b7fd1,
          transparent: true,
          opacity: 0.28,
        }),
      ),
    );

    // Axis lines with arrow heads
    const axisMaterial = new THREE.LineBasicMaterial({
      color: 0xd6cdf6,
      transparent: true,
      opacity: 0.85,
    });
    const axisGeometry = new THREE.BufferGeometry();
    axisGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [
          // key axis along the front edge
          x0 - 0.3,
          0.01,
          z0 + depth + 0.45,
          x0 + width + 0.6,
          0.01,
          z0 + depth + 0.45,
          // query axis along the left edge
          x0 - 0.45,
          0.01,
          z0 + depth + 0.3,
          x0 - 0.45,
          0.01,
          z0 - 0.6,
        ],
        3,
      ),
    );
    rt.content.add(new THREE.LineSegments(axisGeometry, axisMaterial));
    const coneMaterial = new THREE.MeshBasicMaterial({ color: 0xd6cdf6 });
    const keyArrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.09, 0.3, 10),
      coneMaterial,
    );
    keyArrow.rotation.z = -Math.PI / 2;
    keyArrow.position.set(x0 + width + 0.7, 0.01, z0 + depth + 0.45);
    rt.content.add(keyArrow);
    const queryArrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.09, 0.3, 10),
      coneMaterial,
    );
    queryArrow.rotation.x = -Math.PI / 2;
    queryArrow.position.set(x0 - 0.45, 0.01, z0 - 0.7);
    rt.content.add(queryArrow);

    // Tick labels
    const niceStep = (n) => {
      const raw = n / 8;
      return (
        [1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64].find((s) => s >= raw) || 64
      );
    };
    const makeLabel = (text, x, z, options = {}) => {
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 80;
      const context = canvas.getContext("2d");
      const font = `${options.italic ? "italic " : ""}${options.weight || 500} ${options.size || 34}px "Segoe UI", system-ui, sans-serif`;
      context.font = font;
      context.textAlign = "center";
      context.textBaseline = "middle";
      if (options.highlight) {
        const w = context.measureText(String(text)).width + 34;
        context.fillStyle = "#ff3d8f";
        context.beginPath();
        context.roundRect(160 - w / 2, 14, w, 52, 14);
        context.fill();
        context.fillStyle = "#fff6fa";
      } else context.fillStyle = options.color || "#ebe6ff";
      context.fillText(String(text), 160, 40);
      const map = new THREE.CanvasTexture(canvas);
      map.colorSpace = THREE.SRGBColorSpace;
      map.anisotropy = 4;
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map, depthTest: false, transparent: true }),
      );
      sprite.position.set(x, 0.02, z);
      const scale = options.scale || 1;
      sprite.scale.set(2 * scale, 0.5 * scale, 1);
      sprite.renderOrder = 100;
      rt.content.add(sprite);
    };
    const colStep = niceStep(cols),
      rowStep = niceStep(rows);
    const colTicks = new Set();
    for (let c = 0; c < cols; c += colStep) colTicks.add(c);
    colTicks.add(cols - 1);
    const rowTicks = new Set();
    for (let r = 0; r < rows; r += rowStep) rowTicks.add(r);
    rowTicks.add(rows - 1);
    const selectedValue = matrix[selected?.row]?.[selected?.col];
    const hasSelection = Number.isFinite(selectedValue);
    if (hasSelection) {
      // The highlighted tick replaces any neighbour close enough to overlap it.
      const near = (ticks, picked, step) => {
        for (const t of ticks)
          if (t !== picked && Math.abs(t - picked) < step * 0.6)
            ticks.delete(t);
        ticks.add(picked);
      };
      near(colTicks, selected.col, colStep);
      near(rowTicks, selected.row, rowStep);
    }
    const colText = (c) =>
      axisLabels.colLabels?.[c] ?? (axisLabels.colOffset || 0) + c;
    const rowText = (r) =>
      axisLabels.rowLabels?.[r] ?? (axisLabels.rowOffset || 0) + r;
    const labelled = Boolean(axisLabels.colLabels || axisLabels.rowLabels);
    colTicks.forEach((c) =>
      makeLabel(colText(c), cellX(c), z0 + depth + 1.1, {
        highlight: hasSelection && c === selected.col,
        size: labelled ? 30 : 36,
        scale: labelled ? 1.75 : 1.4,
      }),
    );
    rowTicks.forEach((r) =>
      makeLabel(rowText(r), x0 - (labelled ? 1.75 : 1.3), cellZ(r), {
        highlight: hasSelection && r === selected.row,
        size: labelled ? 30 : 36,
        scale: labelled ? 1.75 : 1.4,
      }),
    );
    makeLabel(axisLabels.col, 0, z0 + depth + 2.1, {
      italic: true,
      size: 40,
      scale: 2.1,
    });
    makeLabel(axisLabels.row, x0 - 1.8, z0 - 1.1, {
      italic: true,
      size: 40,
      scale: 2.1,
    });

    if (!cells.length) return;
    const maxAbs = Math.max(
      ...cells.map(([, , value]) => Math.abs(value)),
      1e-12,
    );
    const scale = mode === "attention" ? 1 : maxAbs;
    const mesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.38,
        metalness: 0.08,
      }),
      cells.length,
    );
    const transforms = new THREE.Object3D();
    const color = new THREE.Color();
    const low = new THREE.Color(0xf2efe4);
    const mid = new THREE.Color(0x7fd9c2);
    const positive = new THREE.Color(0x0f9f86);
    const negative = new THREE.Color(0xf16d83);
    const picked = [];
    const inset = cols > 40 || rows > 40 ? 0.74 : 0.66;
    cells.forEach(([row, col, value], index) => {
      const normalized = Math.min(1, Math.abs(value) / scale);
      const height = Math.max(0.03, normalized * 3.2);
      const direction = value < 0 ? -1 : 1;
      transforms.position.set(cellX(col), (direction * height) / 2, cellZ(row));
      transforms.scale.set(cellWidth * inset, height, cellDepth * inset);
      transforms.updateMatrix();
      mesh.setMatrixAt(index, transforms.matrix);
      if (value < 0) color.copy(low).lerp(negative, normalized);
      else if (normalized < 0.5) color.copy(low).lerp(mid, normalized * 2);
      else color.copy(mid).lerp(positive, (normalized - 0.5) * 2);
      if (hasSelection) {
        if (selected.row === row && selected.col === col) color.copy(pink);
        else if (selected.row === row || selected.col === col)
          color.lerp(pink, 0.42);
      }
      mesh.setColorAt(index, color);
      picked.push([row, col]);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    rt.content.add(mesh);
    if (!hasSelection) {
      rt.setPickData(mesh, picked, null);
      return;
    }
    const x = cellX(selected.col);
    const z = cellZ(selected.row);
    const selectedHeight = Math.max(
      0.03,
      Math.min(1, Math.abs(selectedValue) / scale) * 3.2,
    );
    // Translucent pink bands across the selected row and column
    const bandMaterial = new THREE.MeshBasicMaterial({
      color: 0xff3d8f,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const rowBand = new THREE.Mesh(
      new THREE.PlaneGeometry(width + 0.9, cellDepth),
      bandMaterial,
    );
    rowBand.rotation.x = -Math.PI / 2;
    rowBand.position.set(0.15, 0.012, z);
    rt.content.add(rowBand);
    const colBand = new THREE.Mesh(
      new THREE.PlaneGeometry(cellWidth, depth + 0.9),
      bandMaterial,
    );
    colBand.rotation.x = -Math.PI / 2;
    colBand.position.set(x, 0.012, -0.15);
    rt.content.add(colBand);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(
          cellWidth * 1.12,
          selectedHeight + 0.06,
          cellDepth * 1.12,
        ),
      ),
      new THREE.LineBasicMaterial({ color: 0xffd3e6, depthTest: false }),
    );
    outline.position.set(x, selectedHeight / 2, z);
    outline.renderOrder = 90;
    rt.content.add(outline);
    const bandEdges = new THREE.BufferGeometry();
    bandEdges.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [
          x0 - 0.3,
          0.02,
          z - cellDepth / 2,
          x0 + width + 0.6,
          0.02,
          z - cellDepth / 2,
          x0 - 0.3,
          0.02,
          z + cellDepth / 2,
          x0 + width + 0.6,
          0.02,
          z + cellDepth / 2,
          x - cellWidth / 2,
          0.02,
          z0 - 0.6,
          x - cellWidth / 2,
          0.02,
          z0 + depth + 0.3,
          x + cellWidth / 2,
          0.02,
          z0 - 0.6,
          x + cellWidth / 2,
          0.02,
          z0 + depth + 0.3,
        ],
        3,
      ),
    );
    rt.content.add(
      new THREE.LineSegments(
        bandEdges,
        new THREE.LineBasicMaterial({
          color: 0xff6fae,
          transparent: true,
          opacity: 0.85,
          depthTest: false,
        }),
      ),
    );
    rt.setPickData(
      mesh,
      picked,
      new THREE.Vector3(x, Math.max(selectedHeight, 0.03) + 0.1, z),
    );
  }, [
    matrix,
    selected?.row,
    selected?.col,
    mode,
    axisLabels.rowOffset,
    axisLabels.colOffset,
    axisLabels.row,
    axisLabels.col,
    axisLabels.rowLabels,
    axisLabels.colLabels,
  ]);

  return (
    <div
      ref={mount}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        minHeight: 240,
        overflow: "hidden",
      }}
    >
      {failure && (
        <p
          role="status"
          style={{
            position: "absolute",
            inset: "20% 15% auto",
            color: "#f8f1ef",
            zIndex: 2,
          }}
        >
          {failure}
        </p>
      )}
      {callout && !failure && (
        <div ref={calloutRef} className="matrix-callout-anchor">
          <div className="matrix-callout">{callout}</div>
        </div>
      )}
    </div>
  );
}

function disposeGroup(group) {
  const geometries = new Set();
  const materials = new Set();
  group.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.material)
      (Array.isArray(object.material)
        ? object.material
        : [object.material]
      ).forEach((material) => materials.add(material));
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => {
    material.map?.dispose();
    material.dispose();
  });
  group.clear();
}
