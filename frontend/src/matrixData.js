/** Return a bounded, explicitly indexed view of captured tensor data.
 * Missing prefix elements are never interpreted as zeros. Null values were
 * captured but were non-finite. Axis semantics are only used when recorded.
 */
export function extractMatrix(trace, event, options = {}) {
  const attention = trace?.kind === "attention_capture";
  const source = attention ? trace : event;
  if (!source || !Array.isArray(source.shape) || !Array.isArray(source.values))
    return null;
  if (!attention && source.selection !== "flattened_prefix") return null;
  const shape = source.shape;
  if (shape.length > 16 || shape.some((d) => !Number.isSafeInteger(d) || d < 0))
    throw new Error("Invalid tensor shape");
  let total = 1;
  for (const dimension of shape) {
    total *= dimension;
    if (!Number.isSafeInteger(total))
      throw new Error("Tensor shape exceeds safe indexing limits");
  }
  if (attention && shape.length !== 2)
    throw new Error("Attention matrix must have two dimensions");
  const dims =
    shape.length >= 2
      ? shape.slice(-2)
      : shape.length === 1
        ? [1, shape[0]]
        : [1, 1];
  const leading = shape.slice(0, Math.max(0, shape.length - 2));
  const fixed = options.fixedIndices ?? leading.map(() => 0);
  if (
    !Array.isArray(fixed) ||
    fixed.length !== leading.length ||
    fixed.some((n, i) => !Number.isSafeInteger(n) || n < 0 || n >= leading[i])
  )
    throw new Error("Invalid fixed tensor indices");
  const rowOffset = options.rowOffset ?? 0,
    colOffset = options.colOffset ?? 0;
  const maxRows = options.maxRows ?? 128,
    maxCols = options.maxCols ?? 128;
  if (
    ![rowOffset, colOffset, maxRows, maxCols].every(Number.isSafeInteger) ||
    rowOffset < 0 ||
    colOffset < 0 ||
    maxRows < 1 ||
    maxRows > 128 ||
    maxCols < 1 ||
    maxCols > 128 ||
    rowOffset > dims[0] ||
    colOffset > dims[1]
  )
    throw new Error("Invalid matrix window");
  const rows = Math.min(maxRows, dims[0] - rowOffset),
    cols = Math.min(maxCols, dims[1] - colOffset);
  const values = source.values;
  if (
    attention &&
    (values.length !== dims[0] ||
      values.some((row) => !Array.isArray(row) || row.length !== dims[1]))
  )
    throw new Error("Attention values do not match shape");
  if (
    attention &&
    values.some((row) =>
      row.some(
        (value) =>
          value !== null && (!Number.isFinite(value) || value < 0 || value > 1),
      ),
    )
  )
    throw new Error("Attention weights must be between 0 and 1");
  if (
    !attention &&
    (values.length > total ||
      values.some((v) => v !== null && !Number.isFinite(v)))
  )
    throw new Error("Invalid captured prefix");
  let leadingOffset = 0;
  for (let i = 0; i < leading.length; i++)
    leadingOffset = leadingOffset * leading[i] + fixed[i];
  const matrix = [],
    states = [];
  for (let r = 0; r < rows; r++) {
    const row = [],
      rowStates = [];
    for (let c = 0; c < cols; c++) {
      const rr = rowOffset + r,
        cc = colOffset + c;
      const flatIndex = (leadingOffset * dims[0] + rr) * dims[1] + cc;
      const present = attention || flatIndex < values.length;
      const value = present
        ? attention
          ? values[rr][cc]
          : values[flatIndex]
        : null;
      if (value !== null && !Number.isFinite(value))
        throw new Error("Invalid tensor value");
      row.push(value);
      rowStates.push(
        !present ? "uncaptured" : value === null ? "nonfinite" : "captured",
      );
    }
    matrix.push(row);
    states.push(rowStates);
  }
  const axes =
    attention &&
    Array.isArray(source.axes) &&
    source.axes.length === 2 &&
    source.axes.every((a) => typeof a === "string")
      ? source.axes
      : shape.length >= 2
        ? [`dim ${shape.length - 2}`, `dim ${shape.length - 1}`]
        : shape.length === 1
          ? ["single row", "dim 0"]
          : ["scalar", "scalar"];
  return {
    matrix,
    states,
    rows,
    cols,
    shape: [...shape],
    dimensions: dims,
    axes,
    rowOffset,
    colOffset,
    fixedIndices: [...fixed],
    kind: attention ? "attention" : "activation",
    selection: attention ? "full_attention_head" : "flattened_prefix",
    source: attention
      ? `Layer ${source.layer} · head ${source.head}`
      : (source.module ??
        (source.step === undefined ? "Model output" : `Step ${source.step}`)),
    capturedCount: attention ? total : values.length,
    totalElements: total,
  };
}
