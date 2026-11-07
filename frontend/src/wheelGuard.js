/**
 * Let the page scroll over a 3D view. The wheel zooms the view only with Ctrl or ⌘ held (a trackpad
 * pinch arrives as Ctrl + wheel) or after the view is clicked, until the pointer leaves it.
 * Returns a function that removes the listeners.
 */
export function guardWheel(element, controls) {
  let engaged = false;
  const engage = () => {
    engaged = true;
  };
  const release = () => {
    engaged = false;
  };
  // Capture runs before OrbitControls' own wheel listener on the same element.
  const onWheel = (event) => {
    controls.enableZoom = wheelZooms(event, engaged);
  };
  element.addEventListener("pointerdown", engage);
  element.addEventListener("pointerleave", release);
  element.addEventListener("wheel", onWheel, { capture: true, passive: true });
  return () => {
    element.removeEventListener("pointerdown", engage);
    element.removeEventListener("pointerleave", release);
    element.removeEventListener("wheel", onWheel, { capture: true });
  };
}

/** Whether a wheel event should zoom, given whether the view was clicked; exported for tests. */
export const wheelZooms = (event, engaged) =>
  Boolean(engaged || event.ctrlKey || event.metaKey);
