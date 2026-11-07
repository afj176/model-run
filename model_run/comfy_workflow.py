"""ComfyUI workflows as ComfyUI runs them: a saved canvas (the UI format) turned into the prompt /prompt queues.

ComfyUI's frontend does this when you press Run. Model Run does it on the server, so it can run a workflow
template without a ComfyUI tab: subgraphs are flattened (inner nodes become "outer:inner", as the frontend
names them), reroutes and primitive nodes are followed, muted nodes drop out and bypassed ones pass their
input through. Widget values are matched to inputs in the order of each node's definition.

Pure: node definitions are passed in, so this is tested without ComfyUI.
"""

SUBGRAPH_INPUT = -10
SUBGRAPH_OUTPUT = -20
MUTED, BYPASSED = 2, 4
VIRTUAL = {"Reroute", "PrimitiveNode", "Note", "MarkdownNote"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}
SEED_CONTROLS = {"fixed", "increment", "decrement", "randomize"}


class WorkflowError(ValueError):
    pass


def node_definitions():
    """Every installed node's inputs, as ComfyUI's /object_info describes them. Inside ComfyUI only."""
    import nodes
    from comfy_api.internal import _ComfyNodeInternal
    found = {}
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            if issubclass(cls, _ComfyNodeInternal):
                info = cls.GET_NODE_INFO_V1()
                found[name] = {"input": info["input"], "input_order": info.get("input_order") or {k: list(v) for k, v in info["input"].items()},
                               "output": list(info.get("output") or [])}
            else:
                inputs = cls.INPUT_TYPES()
                found[name] = {"input": inputs, "input_order": {k: list(v) for k, v in inputs.items()}, "output": list(getattr(cls, "RETURN_TYPES", ()))}
        except Exception:  # noqa: BLE001 - a broken custom node only matters if a workflow uses it
            continue
    return found


# Types some node outputs, so an input of that type is a connection; set from the definitions in use.
_SOCKETS = set()


def _is_widget(spec):
    """Whether the canvas draws this input as a widget, whose value is saved with the node.

    That is a value type, or a type the canvas draws itself (a colour, a 3D viewport, an image
    comparison): one marked socketless, or one no node outputs, so nothing could connect to it."""
    kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if options.get("forceInput"):
        return False
    if isinstance(kind, list) or kind in WIDGET_TYPES or options.get("socketless"):
        return True
    return bool(_SOCKETS) and isinstance(kind, str) and kind != "*" and "," not in kind and kind not in _SOCKETS


def _ordered(inputs, order):
    """(name, spec) in the definition's order: required, then optional."""
    for section in ("required", "optional"):
        names = (order or {}).get(section) or list((inputs or {}).get(section, {}))
        for name in names:
            spec = (inputs or {}).get(section, {}).get(name)
            if spec is not None:
                yield name, spec


def _default(spec):
    kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in options:
        return options["default"]
    if isinstance(kind, list) and kind:
        return kind[0]
    if kind == "COMBO" and options.get("options"):
        return options["options"][0]
    if kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
        return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[kind]
    return {}  # a widget the canvas draws itself, with nothing saved yet


def widget_values(definition, values, defaults=True):
    """{input name: value} for a node's saved widget values, in its definition's order.

    A widget the saved values do not reach (an input added to the node after the workflow was saved)
    takes its default, as it does on the canvas."""
    if isinstance(values, dict):
        found = dict(values)
        if defaults:
            for name, spec in _ordered(definition.get("input"), definition.get("input_order")):
                if _is_widget(spec) and name not in found and _default(spec) is not None:
                    found[name] = _default(spec)
        return found
    values = list(values or [])
    found = {}
    position = [0]

    def take(name, spec, prefix=""):
        key = f"{prefix}{name}"
        if position[0] >= len(values):
            if defaults and _default(spec) is not None:
                found[key] = value = _default(spec)
            else:
                return
        else:
            value = values[position[0]]
            position[0] += 1
            found[key] = value
        kind = spec[0] if isinstance(spec, (list, tuple)) else spec
        options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
        # A seed may carry the canvas's "control after generate" setting, which is not an input.
        if kind == "INT" and position[0] < len(values) and isinstance(values[position[0]], str) and values[position[0]] in SEED_CONTROLS:
            position[0] += 1
        if kind == "COMFY_DYNAMICCOMBO_V3":
            chosen = next((o for o in options.get("options", []) if o.get("key") == value), None)
            for sub, sub_spec in _ordered((chosen or {}).get("inputs"), None):
                if _is_widget(sub_spec):
                    take(sub, sub_spec, f"{key}.")

    for name, spec in _ordered(definition.get("input"), definition.get("input_order")):
        if _is_widget(spec):
            take(name, spec)
    return found


class _Scope:
    """One graph level: the top canvas, or one placed instance of a subgraph."""

    def __init__(self, nodes, links, prefix, instance=None, parent=None, definition=None):
        self.nodes = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
        self.links = {}
        for link in links or []:
            if isinstance(link, dict):
                self.links[link["id"]] = (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"])
            elif isinstance(link, (list, tuple)) and len(link) >= 5:
                self.links[link[0]] = (link[1], link[2], link[3], link[4])
        self.prefix, self.instance, self.parent, self.definition = prefix, instance, parent, definition


def to_prompt(workflow, definitions):
    """The prompt ComfyUI would queue for this workflow: {node id: {class_type, inputs, _meta}}."""
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        raise WorkflowError("This is not a ComfyUI workflow")
    _SOCKETS.clear()
    _SOCKETS.update(t for d in definitions.values() for t in d.get("output", []) if isinstance(t, str))
    subgraphs = {s["id"]: s for s in (workflow.get("definitions") or {}).get("subgraphs", []) if isinstance(s, dict) and "id" in s}
    prompt = {}
    top = _Scope(workflow["nodes"], workflow.get("links"), "")

    def gid(scope, node_id):
        return f"{scope.prefix}{node_id}"

    def inner_scope(scope, node):
        definition = subgraphs[node["type"]]
        return _Scope(definition.get("nodes", []), definition.get("links", []), f"{gid(scope, node['id'])}:", node, scope, definition)

    def instance_input(scope, index):
        """What feeds input `index` of the subgraph instance that holds this scope: a link or a widget value."""
        node, outer = scope.instance, scope.parent
        name = (scope.definition.get("inputs") or [{}] * (index + 1))[index].get("name") if index < len(scope.definition.get("inputs") or []) else None
        entry = next((i for i in node.get("inputs", []) if i.get("name") == name), None)
        if entry is not None and entry.get("link") is not None:
            return resolve_link(outer, entry["link"])
        # An instance keeps a value for each of the subgraph's widget inputs, in the subgraph's order.
        values = widget_values({"input": {"required": {i.get("name"): [i.get("type")] for i in scope.definition.get("inputs") or []}},
                                "input_order": {"required": [i.get("name") for i in scope.definition.get("inputs") or []]}},
                               node.get("widgets_values"), defaults=False)
        return ("value", values[name]) if name in values else None

    def resolve_output(scope, node_id, slot, depth=0):
        """The real node output behind an output slot: ("link", id, slot), ("value", v) or None."""
        if depth > 64:
            raise WorkflowError("A loop of reroutes or bypassed nodes")
        if node_id == SUBGRAPH_INPUT:
            return instance_input(scope, slot)
        node = scope.nodes.get(node_id)
        if node is None:
            return None
        kind, mode = node.get("type"), node.get("mode", 0)
        if mode == MUTED:
            return None
        if mode == BYPASSED:
            # A bypassed node hands on the input of the output's type, the same slot first.
            out_type = ((node.get("outputs") or [{}] * (slot + 1))[slot] or {}).get("type") if slot < len(node.get("outputs") or []) else None
            candidates = [i for i in node.get("inputs", []) if i.get("link") is not None and (out_type is None or i.get("type") == out_type)]
            same = [i for idx, i in enumerate(node.get("inputs", [])) if idx == slot and i in candidates]
            chosen = (same or candidates or [None])[0]
            return resolve_link(scope, chosen["link"], depth + 1) if chosen else None
        if kind in subgraphs:
            inner = inner_scope(scope, node)
            outputs = inner.definition.get("outputs") or []
            if slot >= len(outputs):
                return None
            for link_id in outputs[slot].get("linkIds") or []:
                link = inner.links.get(link_id)
                if link and link[2] == SUBGRAPH_OUTPUT:
                    return resolve_output(inner, link[0], link[1], depth + 1)
            return None
        if kind == "Reroute":
            entry = (node.get("inputs") or [{}])[0]
            return resolve_link(scope, entry.get("link"), depth + 1)
        if kind == "PrimitiveNode":
            values = node.get("widgets_values") or []
            return ("value", values[0]) if values else None
        return ("link", gid(scope, node_id), slot)

    def resolve_link(scope, link_id, depth=0):
        if link_id is None:
            return None
        link = scope.links.get(link_id)
        return resolve_output(scope, link[0], link[1], depth) if link else None

    def emit(scope):
        for node_id, node in scope.nodes.items():
            kind, mode = node.get("type"), node.get("mode", 0)
            if kind in VIRTUAL or mode in (MUTED, BYPASSED) or node_id in (SUBGRAPH_INPUT, SUBGRAPH_OUTPUT):
                continue
            if kind in subgraphs:
                emit(inner_scope(scope, node))
                continue
            definition = definitions.get(kind)
            if definition is None:
                raise WorkflowError(f"This workflow needs the {kind} node, which is not installed in this ComfyUI")
            # A list value is wrapped, as the canvas does, so ComfyUI does not read it as a connection.
            inputs = {k: {"__value__": v} if isinstance(v, list) else v for k, v in widget_values(definition, node.get("widgets_values")).items()}
            for entry in node.get("inputs", []):
                name = entry.get("name")
                if not name or entry.get("link") is None:
                    continue
                found = resolve_link(scope, entry["link"])
                if found is None:
                    continue
                inputs[name] = found[1] if found[0] == "value" else [found[1], found[2]]
            prompt[gid(scope, node_id)] = {"class_type": kind, "inputs": inputs, "_meta": {"title": node.get("title") or kind}}

    emit(top)
    if not prompt:
        raise WorkflowError("This workflow has no nodes to run")
    return prompt
