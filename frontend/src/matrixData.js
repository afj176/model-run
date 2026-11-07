/** Return a bounded, explicitly indexed view of captured tensor data.
 * Missing prefix elements are never interpreted as zeros. Null values were
 * captured but were non-finite. Axis semantics are only used when recorded.
 */
export function extractMatrix(trace, event, options = {}) {
  const attention = trace?.kind === "attention_capture";
  const source = attention ? trace : event;
  if (!source || !Array.isArray(source.shape) || !Array.isArray(source.values))
    return null;
  if (
    !attention &&
    source.selection !== "flattened_prefix" &&
    source.selection !== "tensor_slice"
  )
    return null;
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
      : source.selection === "tensor_slice" && Array.isArray(source.axes)
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
    selection: attention ? "full_attention_head" : source.selection,
    source: attention
      ? `Layer ${source.layer} · head ${source.head}`
      : source.module ||
        (source.step === undefined ? "Model output" : `Step ${source.step}`),
    capturedCount: attention ? total : values.length,
    totalElements: total,
  };
}

/** Stable key for an axis across events: its name when known, otherwise its position. */
export const axisKey = (axis) => axis.name ?? `axis ${axis.dim}`;
export const axisLabel = axisKey;

/**
 * Position in each axis's recorded indices for the wanted selection. Null means the
 * all-values map of the aggregate axis. A wanted index that was not recorded moves to
 * the nearest one that was, so a selection survives moving between events.
 */
export function resolveSlice(slices, wanted = {}) {
  return slices.axes.map((axis, i) => {
    const want = wanted[axisKey(axis)];
    if (i === slices.aggregate_axis && (want === undefined || want === "all"))
      return null;
    // With guidance, a batch's first entry is usually the negative prompt.
    if (!Number.isInteger(want))
      return Math.max(0, axis.indices.indexOf(axis.conditional));
    let best = 0;
    axis.indices.forEach((index, p) => {
      if (Math.abs(index - want) < Math.abs(axis.indices[best] - want))
        best = p;
    });
    return best;
  });
}

/**
 * The event as seen through one recorded slice: its values and shape become that slice's, so
 * every view reads it like any other 2-D tensor. Events without slices are returned unchanged.
 */
export function sliceEvent(event, wanted = {}) {
  const slices = event?.slices;
  if (!Array.isArray(slices?.axes) || !Array.isArray(slices.values))
    return event;
  const positions = resolveSlice(slices, wanted);
  const all = positions[slices.aggregate_axis] === null;
  let offset = 0;
  slices.axes.forEach((axis, i) => {
    if (positions[i] !== null)
      offset = offset * axis.indices.length + positions[i];
  });
  const size = slices.rows * slices.cols;
  const source = all ? slices.aggregate : slices.values;
  const axisPart = slices.axes
    .map((axis, i) =>
      positions[i] === null
        ? `all ${axisLabel(axis)}s`
        : axis.conditional === axis.indices[positions[i]]
          ? "conditional"
          : `${axisLabel(axis)} ${axis.indices[positions[i]]}`,
    )
    .join(" · ");
  if (slices.layout === "tokens") return tokenGridEvent(event, slices);
  const label =
    slices.layout === "sequence"
      ? [`${slices.row_axis} by ${slices.col_axis}`, axisPart]
          .filter(Boolean)
          .join(" · ")
      : axisPart;
  const notes = [];
  if (all)
    notes.push(
      `Mean magnitude across every ${axisLabel(slices.axes[slices.aggregate_axis])}.`,
    );
  if (slices.pooled)
    notes.push(
      `Averaged from ${slices.source_rows} × ${slices.source_cols} to ${slices.rows} × ${slices.cols}.`,
    );
  for (const axis of slices.axes)
    if (axis.indices.length < axis.size)
      notes.push(
        `${axis.indices.length} of ${axis.size.toLocaleString()} ${axisLabel(axis)}s recorded, evenly spaced.`,
      );
  if (slices.assumption) notes.push(slices.assumption);
  const name = (axis, fallback) => axis ?? fallback;
  return {
    ...event,
    values: source.slice(offset * size, (offset + 1) * size),
    shape: [slices.rows, slices.cols],
    selection: "tensor_slice",
    total_elements: size,
    truncated: false,
    axes: [name(slices.row_axis, "row"), name(slices.col_axis, "column")],
    grid_cols: slices.cols,
    source_shape: event.shape,
    slice_label: label,
    magnitude: all,
    slice_note: notes.join(" "),
  };
}

/** A text layer's token grid: every token (rows) by the same recorded features (columns), exact values. */
export function tokenGridEvent(event, slices) {
  const notes = [
    slices.cols < slices.source_cols
      ? `Every token by ${slices.cols} of its ${slices.source_cols.toLocaleString()} features, evenly spaced; the feature numbers are the columns.`
      : `Every token by all ${slices.source_cols.toLocaleString()} of its features.`,
  ];
  if (slices.rows < slices.source_rows)
    notes.push(
      `The first ${slices.rows} of ${slices.source_rows} tokens; text comes before padding.`,
    );
  return {
    ...event,
    values: slices.values,
    shape: [slices.rows, slices.cols],
    selection: "tensor_slice",
    total_elements: slices.rows * slices.cols,
    truncated: false,
    axes: ["token", "feature"],
    grid_cols: slices.cols,
    source_shape: event.shape,
    slice_label: "every token by feature",
    magnitude: false,
    slice_note: notes.join(" "),
    row_labels: slices.row_labels ?? null,
    col_labels: slices.feature_indices.map((i) => `#${i}`),
    token_rms: slices.token_rms,
    token_max_abs: slices.token_max_abs,
    token_indices: slices.token_indices,
    feature_indices: slices.feature_indices,
    patch_grid: slices.patch_grid ?? null,
  };
}

const MAX_VOLUME_LAYERS = 16;

/**
 * The event as a true volume: recorded maps stacked along one axis, frames first. Tensors with no
 * recorded maps along an axis come back flat, because stacking them would invent a layout.
 */
export function volumeEvent(event, wanted = {}) {
  const base = sliceEvent(event, wanted);
  const slices = event?.slices;
  if (
    !Array.isArray(slices?.axes) ||
    !slices.axes.length ||
    slices.layout === "sequence"
  )
    return base;
  const named = slices.axes.findIndex((a) => a.name === "frame");
  const other = slices.axes.findIndex(
    (a, i) => i !== slices.aggregate_axis && a.indices.length > 1,
  );
  const at = named >= 0 ? named : other >= 0 ? other : slices.aggregate_axis;
  const axis = slices.axes[at];
  if (!axis || axis.indices.length < 2) return base;
  const count = Math.min(MAX_VOLUME_LAYERS, axis.indices.length);
  const chosen = [
    ...new Set(
      Array.from(
        { length: count },
        (_, i) => axis.indices[Math.floor((i * axis.indices.length) / count)],
      ),
    ),
  ];
  const values = [];
  let last = base;
  for (const index of chosen) {
    last = sliceEvent(event, { ...wanted, [axisKey(axis)]: index });
    values.push(...last.values);
  }
  const others = last.slice_label
    .split(" · ")
    .filter((_, i) => i !== at)
    .join(" · ");
  const name = axisLabel(axis);
  return {
    ...last,
    values,
    shape: [slices.rows, slices.cols, chosen.length],
    total_elements: values.length,
    grid_cols: slices.cols,
    slice_label: `${others ? `${others} · ` : ""}stacked by ${name}`,
    volume_layout: {
      rows: slices.rows,
      cols: slices.cols,
      depth: chosen.length,
      axis: name,
      axisKey: axisKey(axis),
      indices: chosen,
      rowAxis: slices.row_axis ?? "row",
      colAxis: slices.col_axis ?? "column",
    },
  };
}

/** Where one value of a volume sits, in words. */
export function volumeCell(layout, index) {
  const plane = layout.rows * layout.cols;
  const within = index % plane;
  return {
    layer: layout.indices[Math.floor(index / plane)],
    row: Math.floor(within / layout.cols),
    col: within % layout.cols,
    text: `${layout.axis} ${layout.indices[Math.floor(index / plane)]}, ${layout.rowAxis} ${Math.floor(within / layout.cols)}, ${layout.colAxis} ${within % layout.cols}`,
  };
}
