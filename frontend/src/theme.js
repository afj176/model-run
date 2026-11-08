// Light or dark chrome. A saved choice wins; otherwise the system setting is followed.
const KEY = "model-run.theme";
const media = () =>
  typeof matchMedia === "function"
    ? matchMedia("(prefers-color-scheme: dark)")
    : null;

export function savedTheme() {
  try {
    const v = localStorage.getItem(KEY);
    return v === "dark" || v === "light" ? v : null;
  } catch {
    return null;
  }
}

/** The theme in effect: the saved one, else the system's. */
export function currentTheme() {
  return savedTheme() ?? (media()?.matches ? "dark" : "light");
}

export function applyTheme(theme = currentTheme()) {
  document.documentElement.dataset.theme = theme;
  return theme;
}

export function setTheme(theme) {
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // A private window keeps nothing; the page still switches.
  }
  return applyTheme(theme);
}

/** Apply the theme before the first paint and follow the system while nothing is saved. */
export function startTheme(onChange) {
  applyTheme();
  const m = media();
  if (!m) return () => {};
  const follow = () => {
    if (!savedTheme()) onChange?.(applyTheme());
  };
  m.addEventListener("change", follow);
  return () => m.removeEventListener("change", follow);
}
