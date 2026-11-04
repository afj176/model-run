"""Display structures from serialized artifacts, with no inferred execution edges."""


def tensor_namespace(report, *, name, max_nodes=10000):
    if type(max_nodes) is not int or not 1 <= max_nodes <= 10000:
        raise ValueError("max_nodes must be between 1 and 10000")
    nodes = {"": {"path": "", "type": "Tensor namespace", "parameters": [], "buffers": []}}
    reports = report.get("shards", [report])
    truncated = False
    for shard in reports:
        for tensor in shard.get("tensors", []):
            parts = tensor["name"].split(".")
            parents = parts[:-1]
            for length in range(1, len(parents) + 1):
                path = ".".join(parents[:length])
                if path not in nodes:
                    if len(nodes) >= max_nodes:
                        truncated = True
                        break
                    nodes[path] = {"path": path, "type": "Tensor namespace", "parameters": [], "buffers": []}
            if truncated:
                break
            path = ".".join(parents)
            nodes[path]["parameters"].append({"name": parts[-1], "shape": tensor["shape"], "dtype": tensor["dtype"]})
        if truncated:
            break
    return {"schema_version": "1.0", "kind": "artifact_tensor_namespace", "name": name,
            "nodes": [nodes[key] for key in sorted(nodes, key=lambda path: [(1, int(part)) if part.isdigit() else (0, part) for part in path.split(".")])], "truncated": truncated,
            "provenance": "Serialized safetensors names and shapes; dot-separated namespace grouping",
            "limitations": ["Groups are tensor-name namespaces, not verified runtime modules or execution order.",
                            "No tensor values have been read. No activations or neuron identities are inferred."]}
