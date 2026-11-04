"""Read ONNX graph metadata without executing a model or loading external weights."""
from pathlib import Path

MAX_GRAPH_BYTES = 128 * 1024 * 1024
MAX_NODES = 100000
MAX_DEPTH = 32


def inspect_onnx(path):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX inspection requires the optional onnx package") from exc
    path = Path(path)
    if path.stat().st_size > MAX_GRAPH_BYTES:
        raise ValueError("ONNX file exceeds the graph inspection limit")
    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Could not parse ONNX model: {exc}") from exc
    graph = model.graph

    def value_info(value):
        result = {"name": value.name}
        if value.type.HasField("tensor_type"):
            tensor = value.type.tensor_type
            result["element_type"] = onnx.TensorProto.DataType.Name(tensor.elem_type)
            if tensor.HasField("shape"):
                result["shape"] = [d.dim_value if d.HasField("dim_value") else d.dim_param if d.HasField("dim_param") else None for d in tensor.shape.dim]
        else:
            result["type"] = "non_tensor_or_unspecified"
        return result

    count = 0

    def inspect_graph(current, scope, depth):
        nonlocal count
        if depth > MAX_DEPTH:
            raise ValueError("ONNX graph nesting exceeds the depth limit")
        count += len(current.node)
        if count > MAX_NODES:
            raise ValueError("ONNX graph exceeds the node limit")
        nodes = []
        for index, node in enumerate(current.node):
            node_id = f"{scope}/node:{index}"
            children = []
            for ai, attribute in enumerate(node.attribute):
                graphs = [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else list(attribute.graphs) if attribute.type == onnx.AttributeProto.GRAPHS else []
                for gi, child in enumerate(graphs):
                    child_scope = f"{node_id}/attribute:{ai}/graph:{gi}"
                    children.append({"attribute": attribute.name, "graph": inspect_graph(child, child_scope, depth + 1)})
            nodes.append({"id": node_id, "name": node.name, "op_type": node.op_type,
                          "domain": node.domain, "inputs": list(node.input), "outputs": list(node.output),
                          "subgraphs": children})
        return {"scope": scope, "graph_name": current.name,
                "inputs": [value_info(item) for item in current.input],
                "outputs": [value_info(item) for item in current.output],
                "value_info": [value_info(item) for item in current.value_info],
                "initializers": [{"name": item.name, "shape": list(item.dims),
                                  "element_type": onnx.TensorProto.DataType.Name(item.data_type),
                                  "external": item.data_location == onnx.TensorProto.EXTERNAL} for item in current.initializer],
                "nodes": nodes}

    parsed_graph = inspect_graph(graph, "graph:0", 0)
    return {"schema_version": "1.0", "kind": "onnx_graph", "path": str(path),
            "provenance": "serialized ONNX graph", **parsed_graph,
            "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "limitations": ["External tensor data is not loaded; no model execution or validation is performed.",
                            "Node input/output names are scoped to their graph; nested graphs can refer to outer-scope values.",
                            "Node attributes other than subgraphs, sparse initializers and local function bodies are not expanded; runtime optimizations may differ."],
            "capabilities": {"inventory": {"status": "partial", "reason": "Graph metadata only; external data is not loaded."},
                             "structure": {"status": "partial", "reason": "Serialized graph structure; local functions and attributes are not expanded."},
                             "execution": {"status": "unsupported", "reason": "No runtime adapter is installed."},
                             "observation": {"status": "unsupported", "reason": "No instrumented runtime is installed."}}}
