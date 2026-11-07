import unittest

from model_run.comfy_instrument import drop_optional, instrument, media_inputs, read_settings
from model_run.comfy_workflow import WorkflowError, to_prompt, widget_values

COMBO = [["euler", "dpmpp_2m"]]
DEFINITIONS = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"]]}}, "output": ["MODEL", "CLIP", "VAE"]},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}}, "output": ["CONDITIONING"]},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {"control_after_generate": True}], "steps": ["INT", {"default": 20}],
                                        "cfg": ["FLOAT", {"default": 8.0}], "sampler_name": COMBO, "positive": ["CONDITIONING"],
                                        "negative": ["CONDITIONING"], "latent_image": ["LATENT"], "denoise": ["FLOAT", {"default": 1.0}]}},
                 "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name", "positive", "negative", "latent_image", "denoise"]},
                 "output": ["LATENT"]},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT"], "height": ["INT"], "batch_size": ["INT", {"default": 1}]}}, "output": ["LATENT"]},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}, "output": ["IMAGE"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}}, "output": []},
    "Painter": {"input": {"required": {"mask": ["MASK"], "bg_color": ["COLOR", {"socketless": True}]}}, "output": ["IMAGE"]},
    "LoadImageMask": {"input": {"required": {"image": [["a.png"]]}}, "output": ["MASK"]},
    "TextGenerate": {"input": {"required": {"clip": ["CLIP"], "prompt": ["STRING"], "max_length": ["INT"],
                                            "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                                                {"key": "on", "inputs": {"required": {"temperature": ["FLOAT"], "seed": ["INT"]}}},
                                                {"key": "off", "inputs": {}}]}]},
                               "optional": {"thinking": ["BOOLEAN"]}},
                     "input_order": {"required": ["clip", "prompt", "max_length", "sampling_mode"], "optional": ["thinking"]},
                     "output": ["STRING"]},
}


def node(id, type, inputs=(), widgets=(), mode=0, outputs=()):
    return {"id": id, "type": type, "mode": mode, "inputs": [dict(i) for i in inputs], "outputs": [dict(o) for o in outputs], "widgets_values": list(widgets)}


def text_to_image():
    nodes = [
        node(1, "CheckpointLoaderSimple", widgets=["a.safetensors"]),
        node(2, "CLIPTextEncode", [{"name": "clip", "link": 2}], ["a cat"]),
        node(3, "CLIPTextEncode", [{"name": "clip", "link": 3}], ["blurry"]),
        node(4, "EmptyLatentImage", widgets=[512, 768, 1]),
        node(5, "KSampler", [{"name": "model", "link": 1}, {"name": "positive", "link": 4}, {"name": "negative", "link": 5}, {"name": "latent_image", "link": 6}],
             [42, "randomize", 20, 7.5, "euler", 1.0]),
        node(6, "VAEDecode", [{"name": "samples", "link": 7}, {"name": "vae", "link": 8}]),
        node(7, "SaveImage", [{"name": "images", "link": 9}], ["out"]),
        node(8, "MarkdownNote", widgets=["# notes"]),
    ]
    links = [[1, 1, 0, 5, 0, "MODEL"], [2, 1, 1, 2, 0, "CLIP"], [3, 1, 1, 3, 0, "CLIP"], [4, 2, 0, 5, 1, "CONDITIONING"],
             [5, 3, 0, 5, 2, "CONDITIONING"], [6, 4, 0, 5, 3, "LATENT"], [7, 5, 0, 6, 0, "LATENT"], [8, 1, 2, 6, 1, "VAE"], [9, 6, 0, 7, 0, "IMAGE"]]
    return {"nodes": nodes, "links": links}


class Conversion(unittest.TestCase):
    def test_a_canvas_becomes_the_prompt_comfy_queues(self):
        prompt = to_prompt(text_to_image(), DEFINITIONS)
        self.assertNotIn("8", prompt)  # notes are not run
        self.assertEqual(prompt["5"]["inputs"], {"model": ["1", 0], "seed": 42, "steps": 20, "cfg": 7.5, "sampler_name": "euler",
                                                 "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "denoise": 1.0})
        self.assertEqual(prompt["2"]["inputs"], {"text": "a cat", "clip": ["1", 1]})
        self.assertEqual(prompt["4"]["inputs"], {"width": 512, "height": 768, "batch_size": 1})

    def test_muted_nodes_drop_out_and_bypassed_ones_pass_their_input_on(self):
        workflow = text_to_image()
        workflow["nodes"].append(node(9, "VAEDecode", [{"name": "samples", "link": 7}, {"name": "vae", "link": 8}], mode=2))
        workflow["nodes"].append(node(10, "CLIPTextEncode", [{"name": "clip", "link": None}], ["x"], mode=4,
                                      outputs=[{"type": "CONDITIONING"}]))
        workflow["nodes"][5]["inputs"][0]["link"] = 10  # samples now come through a bypassed sampler copy
        workflow["nodes"].append(node(11, "KSampler", [{"name": "model", "link": 1}, {"name": "latent_image", "link": 11, "type": "LATENT"}],
                                      [1, 20, 8, "euler", 1], mode=4, outputs=[{"type": "LATENT"}]))
        workflow["nodes"][-1]["inputs"][1]["type"] = "LATENT"
        workflow["links"] += [[10, 11, 0, 6, 0, "LATENT"], [11, 5, 0, 11, 7, "LATENT"]]
        prompt = to_prompt(workflow, DEFINITIONS)
        self.assertNotIn("9", prompt)
        self.assertNotIn("10", prompt)
        self.assertNotIn("11", prompt)
        self.assertEqual(prompt["6"]["inputs"]["samples"], ["5", 0])

    def test_reroutes_and_primitive_nodes_are_followed(self):
        workflow = text_to_image()
        workflow["nodes"].append(node(20, "Reroute", [{"name": "", "link": 20}]))
        workflow["nodes"].append(node(21, "PrimitiveNode", widgets=[7], outputs=[{"type": "INT"}]))
        workflow["nodes"][4]["inputs"].append({"name": "seed", "link": 22, "widget": {"name": "seed"}})
        workflow["nodes"][5]["inputs"][1]["link"] = 21
        workflow["links"] += [[20, 1, 2, 20, 0, "VAE"], [21, 20, 0, 6, 1, "VAE"], [22, 21, 0, 5, 4, "INT"]]
        prompt = to_prompt(workflow, DEFINITIONS)
        self.assertEqual(prompt["6"]["inputs"]["vae"], ["1", 2])
        self.assertEqual(prompt["5"]["inputs"]["seed"], 7)
        self.assertNotIn("20", prompt)

    def test_a_subgraph_is_flattened_with_its_own_values(self):
        inner = {"id": "sub-1", "name": "Sampler group",
                 "inputs": [{"name": "model", "type": "MODEL"}, {"name": "seed", "type": "INT"}, {"name": "steps", "type": "INT"}],
                 "outputs": [{"name": "LATENT", "linkIds": [33]}],
                 "nodes": [node(30, "EmptyLatentImage", widgets=[64, 64, 1]),
                           node(31, "KSampler", [{"name": "model", "link": 30}, {"name": "latent_image", "link": 32},
                                                 {"name": "seed", "link": 31, "widget": {"name": "seed"}}, {"name": "steps", "link": 34, "widget": {"name": "steps"}}],
                                [0, "fixed", 5, 1.0, "euler", 1.0])],
                 "links": [{"id": 30, "origin_id": -10, "origin_slot": 0, "target_id": 31, "target_slot": 0, "type": "MODEL"},
                           {"id": 31, "origin_id": -10, "origin_slot": 1, "target_id": 31, "target_slot": 4, "type": "INT"},
                           {"id": 34, "origin_id": -10, "origin_slot": 2, "target_id": 31, "target_slot": 5, "type": "INT"},
                           {"id": 32, "origin_id": 30, "origin_slot": 0, "target_id": 31, "target_slot": 3, "type": "LATENT"},
                           {"id": 33, "origin_id": 31, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "LATENT"}]}
        workflow = {"nodes": [node(1, "CheckpointLoaderSimple", widgets=["a.safetensors"]),
                              # The instance lists only some inputs; its values follow the subgraph's own input order.
                              node(40, "sub-1", [{"name": "model", "link": 1}, {"name": "steps", "link": None, "widget": {"name": "steps"}}], [123, 9]),
                              node(6, "VAEDecode", [{"name": "samples", "link": 2}, {"name": "vae", "link": 3}])],
                    "links": [[1, 1, 0, 40, 0, "MODEL"], [2, 40, 0, 6, 0, "LATENT"], [3, 1, 2, 6, 1, "VAE"]],
                    "definitions": {"subgraphs": [inner]}}
        prompt = to_prompt(workflow, DEFINITIONS)
        self.assertEqual(set(prompt), {"1", "40:30", "40:31", "6"})
        sampler = prompt["40:31"]["inputs"]
        self.assertEqual((sampler["model"], sampler["latent_image"], sampler["seed"], sampler["steps"]), (["1", 0], ["40:30", 0], 123, 9))
        self.assertEqual(prompt["6"]["inputs"]["samples"], ["40:31", 0])

    def test_widget_values_follow_the_definition_and_its_options(self):
        values = widget_values(DEFINITIONS["TextGenerate"], ["hi", 99, "on", 0.3, 5, True])
        self.assertEqual(values, {"prompt": "hi", "max_length": 99, "sampling_mode": "on", "sampling_mode.temperature": 0.3,
                                  "sampling_mode.seed": 5, "thinking": True})
        # An input added after the workflow was saved takes its default, as on the canvas.
        self.assertEqual(widget_values(DEFINITIONS["KSampler"], [1, 2, 3.0, "euler"])["denoise"], 1.0)

    def test_canvas_drawn_inputs_and_list_values(self):
        prompt = to_prompt({"nodes": [node(1, "Painter", widgets=["#ff0000"]), node(2, "SaveImage", widgets=[["a", "b"]])], "links": []}, DEFINITIONS)
        self.assertEqual(prompt["1"]["inputs"]["bg_color"], "#ff0000")
        self.assertEqual(prompt["2"]["inputs"]["filename_prefix"], {"__value__": ["a", "b"]})  # not read as a link

    def test_a_missing_node_type_says_which(self):
        with self.assertRaises(WorkflowError) as missing:
            to_prompt({"nodes": [node(1, "SomeCustomNode")], "links": []}, DEFINITIONS)
        self.assertIn("SomeCustomNode", str(missing.exception))


class Instrumenting(unittest.TestCase):
    def setUp(self):
        self.prompt = to_prompt(text_to_image(), DEFINITIONS)

    def test_the_sampler_is_observed_and_the_text_and_vae_record(self):
        done = instrument(self.prompt)
        prompt = done["prompt"]
        observe = prompt["5"]["inputs"]["model"][0]
        self.assertEqual(prompt[observe]["class_type"], "ModelRunObserve")
        self.assertEqual(prompt[observe]["inputs"]["model"], ["1", 0])
        self.assertEqual(prompt[observe]["inputs"]["attention_block"], "auto")
        self.assertEqual((prompt[observe]["inputs"]["clip"], prompt[observe]["inputs"]["prompt"]), (["1", 1], "a cat"))
        self.assertEqual(prompt["2"]["class_type"], "ModelRunTextEncode")
        self.assertEqual(prompt["6"]["class_type"], "ModelRunVAEDecode")
        attach = next(n for n in prompt.values() if n["class_type"] == "ModelRunAttachOutput")
        self.assertEqual(attach["inputs"], {"observed_model": [observe, 0], "images": ["6", 0]})
        self.assertEqual(self.prompt["5"]["inputs"]["model"], ["1", 0])  # the original is untouched
        self.assertTrue(done["observed"])

    def test_two_samplers_on_one_model_share_an_observer(self):
        self.prompt["20"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0]}}
        prompt = instrument(self.prompt)["prompt"]
        self.assertEqual(prompt["5"]["inputs"]["model"], prompt["20"]["inputs"]["model"])

    def test_settings_reach_the_graph(self):
        self.assertEqual(read_settings(self.prompt), {"prompt": "a cat", "seed": 42, "steps": 20, "width": 512, "height": 768})
        prompt = instrument(self.prompt, {"prompt": "a lighthouse", "seed": 7, "steps": 4, "width": 640, "height": 640})["prompt"]
        self.assertEqual(prompt["2"]["inputs"]["text"], "a lighthouse")
        self.assertEqual(prompt["3"]["inputs"]["text"], "blurry")
        self.assertEqual((prompt["5"]["inputs"]["seed"], prompt["5"]["inputs"]["steps"]), (7, 4))
        self.assertEqual((prompt["4"]["inputs"]["width"], prompt["4"]["inputs"]["height"]), (640, 640))

    def test_generate_text_becomes_model_runs(self):
        source = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "g.safetensors"}},
                  "2": {"class_type": "TextGenerate", "inputs": {"clip": ["1", 0], "prompt": "hi", "max_length": 99, "sampling_mode": "on",
                                                                "sampling_mode.temperature": 0.3, "sampling_mode.seed": 7}}}
        node2 = instrument(source, {"prompt": "hello"})["prompt"]["2"]
        self.assertEqual(node2["class_type"], "ModelRunTextGenerate")
        self.assertEqual((node2["inputs"]["prompt"], node2["inputs"]["sampling"], node2["inputs"]["temperature"], node2["inputs"]["seed"]), ("hello", True, 0.3, 7))

    def test_files_a_workflow_reads(self):
        definitions = {"TextGenerate": {"input": {"required": {"clip": ["CLIP"]}, "optional": {"image": ["IMAGE"], "video": ["IMAGE"]}}},
                       "ImageScale": {"input": {"required": {"image": ["IMAGE"]}}}, "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
                       "GetVideoComponents": {"input": {"required": {"video": ["VIDEO"]}}}}
        prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "example.png"}, "_meta": {"title": "Photo"}},
                  "2": {"class_type": "TextGenerate", "inputs": {"clip": ["9", 0], "image": ["1", 0]}},
                  "3": {"class_type": "LoadImage", "inputs": {"image": "other.png"}},
                  "4": {"class_type": "ImageScale", "inputs": {"image": ["3", 0]}},
                  "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
                  "6": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
                  "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["6", 0]}}}
        prompt["2"]["inputs"]["video"] = ["7", 0]
        self.assertEqual(media_inputs(prompt)[0], {"node": "1", "kind": "image", "file": "example.png", "title": "Photo"})
        # A missing file that only feeds optional inputs is left out, with any node that only processes it;
        # one a saved result depends on is reported.
        self.assertEqual(drop_optional(prompt, ["1", "3", "6"], definitions), ["3"])
        self.assertNotIn("1", prompt)
        self.assertNotIn("7", prompt)
        self.assertNotIn("image", prompt["2"]["inputs"])
        self.assertNotIn("video", prompt["2"]["inputs"])
        chosen = instrument(prompt, {"files": {"3": "mine.png", "4": "../../x.png"}})["prompt"]
        self.assertEqual(chosen["3"]["inputs"]["image"], "mine.png")

    def test_nothing_to_record_is_said_so(self):
        self.assertFalse(instrument({"1": {"class_type": "LoadImage", "inputs": {}}})["observed"])


if __name__ == "__main__":
    unittest.main()
