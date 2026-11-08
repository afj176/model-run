/**
 * Math.max and Math.min over lists of any length. Spreading a list into Math.max overflows the call stack
 * past about a hundred thousand values, which a large capture easily holds. Arguments may be numbers or
 * lists of numbers; the result matches Math.max(...) and Math.min(...) on the same values, NaN included.
 */
function fold(parts, better, start) {
  let best = start;
  for (const part of parts) {
    const list = Array.isArray(part) ? part : [part];
    for (let i = 0; i < list.length; i++) {
      const n = +list[i];
      if (Number.isNaN(n)) return NaN;
      if (better(n, best)) best = n;
    }
  }
  return best;
}
export const mostOf = (...parts) => fold(parts, (a, b) => a > b, -Infinity);
export const leastOf = (...parts) => fold(parts, (a, b) => a < b, Infinity);
