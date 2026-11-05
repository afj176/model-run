"""Check Model Run's ComfyUI node against every model family ComfyUI ships, without any weights.

Builds each family's real ComfyUI class at a tiny size with random weights and runs the real
observer on it. It answers what a person copying a model into ComfyUI needs to know: does observing
change the output, are all layers found, can attention be reached, is the geometry understood, and
where do text keys really sit? That last check changes only the text and sees which keys change.

It tests structure and code paths, not results. Random weights say nothing about what a real model
learned, and tiny configs can differ from released ones.

Run from a ComfyUI folder with its Python, then read the JSON lines or the summary:

    PYTHONPATH=. .venv/Scripts/python.exe <repo>/tools/check_comfy_families.py
"""
import dataclasses, importlib, inspect, json, os, re, sys, traceback
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_run.comfy_bridge import BYPASS_ATTENTION_MODELS, reach_attention, DenoiserObserver, all_layer_paths, all_layer_plan
from model_run.trace_store import encode_trace
ops = comfy.ops.disable_weight_init
TEXT, DIM = 6, 16
def ctx(dim=DIM, n=TEXT): return torch.randn(1, n, dim)
IMG = lambda c=4, s=16: torch.randn(1, c, s, s)
VID = lambda c=16, f=3, s=16: torch.randn(1, c, f, s, s)
FLUX = dict(in_channels=16, out_channels=16, vec_in_dim=8, context_in_dim=DIM, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=False)
HY = dict(in_channels=16, out_channels=16, vec_in_dim=8, context_in_dim=DIM, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=256, patch_size=[1, 2, 2], qkv_bias=True, guidance_embed=False)
CHROMA = dict(FLUX, in_dim=16, out_dim=16) ;
WAN = dict(dim=32, ffn_dim=64, freq_dim=16, text_dim=DIM, num_heads=2, num_layers=2, text_len=TEXT, in_dim=16, out_dim=16)
FAMILIES = [
 ("SD1.5 / SD2 UNet", "comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel", dict(image_size=8, in_channels=4, model_channels=32, out_channels=4, num_res_blocks=[1, 1], channel_mult=[1, 2], num_heads=2, use_spatial_transformer=True, transformer_depth=[1, 1], transformer_depth_output=[1, 1, 1, 1], transformer_depth_middle=1, context_dim=DIM, use_linear_in_transformer=False, legacy=False), IMG(), ctx(), {}),
 ("SDXL UNet", "comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel", dict(image_size=8, in_channels=4, model_channels=32, out_channels=4, num_res_blocks=[1, 1], channel_mult=[1, 2], num_head_channels=8, use_spatial_transformer=True, transformer_depth=[0, 2], transformer_depth_output=[0, 0, 2, 2], transformer_depth_middle=2, context_dim=DIM, use_linear_in_transformer=True, adm_in_channels=8, num_classes="sequential", legacy=False), IMG(), ctx(), {"y": torch.randn(1, 8)}),
 ("Stable Diffusion 3", "comfy.ldm.modules.diffusionmodules.mmdit.OpenAISignatureMMDITWrapper", dict(input_size=16, patch_size=2, in_channels=16, depth=2, num_blocks=2, adm_in_channels=8, out_channels=16, pos_embed_max_size=32, context_embedder_config={"target": "torch.nn.Linear", "params": {"in_features": DIM, "out_features": 128}}, qk_norm="rms", x_block_self_attn=False), IMG(16), ctx(), {"y": torch.randn(1, 8)}),
 ("AuraFlow", "comfy.ldm.aura.mmdit.MMDiT", dict(in_channels=4, out_channels=4, patch_size=2, dim=32, n_layers=3, n_double_layers=1, n_heads=2, global_conddim=32, cond_seq_dim=DIM, max_seq=256), IMG(), ctx(), {}),
 ("Hunyuan DiT", "comfy.ldm.hydit.models.HunYuanDiT", dict(input_size=(16, 16), patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=2, text_states_dim=DIM, text_states_dim_t5=DIM, text_len=TEXT, text_len_t5=TEXT, log_fn=lambda *_: None), IMG(), ctx(), {"encoder_hidden_states_t5": ctx(), "text_embedding_mask": torch.ones(1, TEXT), "text_embedding_mask_t5": torch.ones(1, TEXT)}),
 ("PixArt", "comfy.ldm.pixart.pixartms.PixArtMS", dict(input_size=16, patch_size=2, in_channels=4, hidden_size=32, depth=2, num_heads=2, caption_channels=DIM, model_max_length=TEXT, micro_condition=False, pe_interpolation=1.0, pe_precision=None), IMG(4, 64), ctx(), {"c_size": torch.tensor([[256.0, 256.0]]), "c_ar": torch.tensor([[1.0]])}),
 ("Flux 1", "comfy.ldm.flux.model.Flux", dict(FLUX, txt_ids_dims=[], global_modulation=False, mlp_silu_act=False, ops_bias=True, default_ref_method="offset", ref_index_scale=1.0, yak_mlp=False, txt_norm=False), IMG(16), ctx(), {"y": torch.randn(1, 8)}),
 ("Flux 2", "comfy.ldm.flux.model.Flux", dict(FLUX, global_modulation=True, mlp_silu_act=True, ops_bias=False, default_ref_method="index_timestep_zero", ref_index_scale=1.0, yak_mlp=False, vec_in_dim=None, txt_ids_dims=[3], axes_dim=[4, 4, 4, 4], txt_norm=False), IMG(16), ctx(), {}),
 ("Chroma", "comfy.ldm.chroma.model.Chroma", dict(in_channels=64, out_channels=64, context_in_dim=DIM, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=2, qkv_bias=True, in_dim=64, out_dim=32, hidden_dim=16, n_layers=2, txt_ids_dims=[], vec_in_dim=8), IMG(16), ctx(), {"guidance": torch.zeros(1)}),
 ("Hunyuan Video", "comfy.ldm.hunyuan_video.model.HunyuanVideo", dict(HY, image_model=None, byt5=False, meanflow=False, use_cond_type_embedding=False, vision_in_dim=None, meanflow_sum=False), VID(), ctx(), {"y": torch.randn(1, 8)}),
 ("Wan 2.x", "comfy.ldm.wan.model.WanModel", dict(dim=32, ffn_dim=64, freq_dim=16, text_dim=DIM, num_heads=2, num_layers=2, text_len=TEXT, in_dim=16, out_dim=16), VID(), ctx(), {}),
 ("LTX Video", "comfy.ldm.lightricks.model.LTXVModel", dict(in_channels=128, cross_attention_dim=32, attention_head_dim=8, num_attention_heads=4, caption_channels=DIM, num_layers=2), VID(128), ctx(), {"attention_mask": None}),
 ("Mochi", "comfy.ldm.genmo.joint_model.asymm_models_joint.AsymmDiTJoint", dict(patch_size=2, in_channels=12, hidden_size_x=1536, hidden_size_y=64, depth=2, num_heads=2, qk_norm=True, use_t5=True, t5_feat_dim=DIM, t5_token_length=TEXT, learn_sigma=False), VID(12), ctx(), {"attention_mask": torch.ones(1, TEXT), "num_tokens": TEXT}),
 ("Cosmos Predict2", "comfy.ldm.cosmos.predict2.MiniTrainDIT", dict(max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=32, num_blocks=2, num_heads=2, crossattn_emb_channels=DIM, pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=16), VID(), ctx(), {}),
 ("Lumina 2 / Z-Image", "comfy.ldm.lumina.model.NextDiT", dict(patch_size=2, in_channels=16, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2, cap_feat_dim=DIM, axes_dims=(4, 6, 6), axes_lens=(64, 64, 64)), IMG(16), ctx(), {"num_tokens": TEXT}),
 ("Qwen-Image", "comfy.ldm.qwen_image.model.QwenImageTransformer2DModel", dict(patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=DIM, axes_dims_rope=(4, 6, 6)), VID(16, 1), ctx(), {}),
 ("Qwen-Image 2.1", "comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel", dict(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=2, context_in_dim=DIM, axes_dims_rope=(4, 6, 6)), IMG(64), ctx(), {}),
 ("HiDream I1", "comfy.ldm.hidream.model.HiDreamImageTransformer2DModel", dict(patch_size=2, in_channels=16, out_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2, caption_channels=[DIM, DIM], text_emb_dim=8, axes_dims_rope=(4, 6, 6), llama_layers=[0, 1], num_routed_experts=2, num_activated_experts=1), IMG(16), ctx(DIM), {"y": torch.randn(1, 8), "encoder_hidden_states_llama3": torch.randn(1, 48, TEXT, DIM)}),
 ("Flux 3", "comfy.ldm.flux3.model.Flux3", dict(hidden_size=256, num_heads=2, depth=1, depth_single_blocks=1), IMG(128), ctx(20480), {}),
 ("Lens", "comfy.ldm.lens.model.LensTransformer2DModel", dict(patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=2, enc_hidden_dim=DIM, axes_dims_rope=(4, 6, 6), selected_layer_index=(0,), multi_layer_encoder_feature=False), IMG(64), ctx(), {}),
 ("OmniGen 2", "comfy.ldm.omnigen.omnigen2.OmniGen2Transformer2DModel", dict(patch_size=2, in_channels=16, hidden_size=32, num_layers=2, num_refiner_layers=1, num_attention_heads=2, num_kv_heads=2, axes_dim_rope=(4, 6, 6), axes_lens=(64, 64, 64), text_feat_dim=DIM), IMG(16), ctx(), {"num_tokens": TEXT}),
 ("Boogu", "comfy.ldm.boogu.model.BooguTransformer2DModel", dict(patch_size=2, in_channels=16, hidden_size=32, num_layers=2, num_double_stream_layers=1, num_refiner_layers=1, num_attention_heads=2, num_kv_heads=2, axes_dim_rope=(4, 6, 6), axes_lens=(64, 64, 64), instruction_feat_dim=DIM), IMG(16), ctx(), {"num_tokens": TEXT}),
 ("Stable Audio", "comfy.ldm.audio.dit.AudioDiffusionTransformer", dict(io_channels=8, embed_dim=128, cond_token_dim=DIM, project_cond_tokens=True, global_cond_dim=8, depth=2, num_heads=2), torch.randn(1, 8, 16), ctx(), {"global_embed": torch.randn(1, 8)}),
 ("ACE-Step", "comfy.ldm.ace.model.ACEStepTransformer2DModel", dict(in_channels=8, num_layers=2, inner_dim=32, attention_head_dim=8, num_attention_heads=4, out_channels=8, text_embedding_dim=DIM, ssl_encoder_depths=[1, 1], speaker_embedding_dim=8), torch.randn(1, 8, 16, 16), ctx(), {"attention_mask": torch.ones(1, 16), "text_attention_mask": torch.ones(1, TEXT), "speaker_embeds": torch.randn(1, 8), "lyric_token_idx": torch.zeros(1, 4, dtype=torch.long), "lyric_mask": torch.ones(1, 4)}),
 ("Mage Flow", "comfy.ldm.mage_flow.model.MageFlowTransformer2DModel", dict(in_channels=64, out_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=DIM, axes_dims_rope=(4, 6, 6)), IMG(64), ctx(), {}),
 ("JoyImage", "comfy.ldm.joyimage.model.JoyImageTransformer3DModel", dict(patch_size=[1, 2, 2], in_channels=16, hidden_size=32, num_attention_heads=2, text_dim=DIM, num_layers=2, rope_dim_list=[4, 6, 6]), VID(16, 1), ctx(), {}),
 ("Ideogram 4", "comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel", dict(in_channels=64, num_layers=2, num_attention_heads=2, attention_head_dim=16, intermediate_size=64, adaln_dim=16, llm_features_dim=DIM, mrope_section=(4, 2, 2)), IMG(64), ctx(), {}),
 ("Krea 2", "comfy.ldm.krea2.model.SingleStreamDiT", dict(features=32, tdim=16, txtdim=DIM, heads=2, kvheads=2, multiplier=2, layers=2, patch=2, channels=16, txtlayers=1, txtheads=2, txtkvheads=2), IMG(16), ctx(), {}),
 ("MOSS Image", "comfy.ldm.moss_image.model.MOSSImage", dict(dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=DIM, latent_channels=64), IMG(64), ctx(), {}),
 ("Fibo 2", "comfy.ldm.fibo2.model.Fibo2", dict(in_channels=64, dim=256, n_heads=2, n_layers=2, n_refiner_layers=1, injection_layers=(0,), cap_feat_dim=DIM, perceiver_layers=1, max_gist_tokens=256, ffn_dim=64), IMG(64), ctx(2 * DIM), {}),
 ("Kandinsky 5", "comfy.ldm.kandinsky5.model.Kandinsky5", dict(in_visual_dim=16, out_visual_dim=16, in_text_dim=DIM, in_text_dim2=8, time_dim=16, model_dim=32, ff_dim=64, visual_embed_dim=64, num_text_blocks=1, num_visual_blocks=2, axes_dims=(4, 6, 6)), VID(), ctx(), {"y": torch.randn(1, 8)}),
 ("Ernie Image", "comfy.ldm.ernie.model.ErnieImageModel", dict(hidden_size=32, num_attention_heads=2, num_layers=2, ffn_hidden_size=64, in_channels=16, out_channels=16, patch_size=2, text_in_dim=DIM, rope_axes_dim=(4, 6, 6)), IMG(16), ctx(), {}),
 ("CogVideoX", "comfy.ldm.cogvideo.model.CogVideoXTransformer3DModel", dict(num_attention_heads=2, attention_head_dim=24, in_channels=16, out_channels=16, time_embed_dim=16, text_embed_dim=DIM, num_layers=2, sample_width=16, sample_height=16, sample_frames=9, max_text_seq_length=TEXT), VID(16, 3), ctx(), {}),
 ("Hunyuan Image 2.1", "comfy.ldm.hunyuan_video.model.HunyuanVideo", dict(HY, image_model=None, patch_size=[2, 2], axes_dim=[8, 8], byt5=False, meanflow=False, use_cond_type_embedding=False, vision_in_dim=None, meanflow_sum=False), IMG(16), ctx(), {"y": torch.randn(1, 8)}),
 ("PixelDiT", "comfy.ldm.pixeldit.model.PixDiT_T2I", dict(in_channels=3, num_groups=2, hidden_size=32, pixel_hidden_size=8, pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=2, pixel_depth=1, patch_size=4, txt_embed_dim=DIM, txt_max_length=TEXT), IMG(3), ctx(), {}),
 ("Wan causal AR", "comfy.ldm.wan.ar_model.CausalWanModel", WAN, VID(), ctx(), {}),
 ("Wan VACE", "comfy.ldm.wan.model.VaceWanModel", dict(WAN, vace_layers=1, vace_in_dim=32), VID(), ctx(), {"vace_context": torch.randn(1, 1, 32, 3, 16, 16), "vace_strength": [1.0]}),
 ("Wan Camera", "comfy.ldm.wan.model.CameraWanModel", dict(WAN), VID(), ctx(), {}),
 ("Wan HuMo", "comfy.ldm.wan.model.HumoWanModel", dict(WAN), VID(), ctx(), {}),
 ("Wan Animate", "comfy.ldm.wan.model_animate.AnimateWanModel", dict(WAN, in_dim=16), VID(), ctx(), {}),
 ("Wan Animate 2", "comfy.ldm.wan.model_animate2.WanAnimate2Model", dict(WAN), VID(), ctx(), {}),
 ("Wan S2V", "comfy.ldm.wan.model.WanModel_S2V", dict(WAN, audio_inject_layers=[0], cond_dim=16, audio_dim=8), VID(), ctx(), {}),
 ("Wan SCAIL", "comfy.ldm.wan.model.SCAILWanModel", dict(WAN, in_dim=16), VID(), ctx(), {}),
 ("Wan SCAIL 2", "comfy.ldm.wan.model.SCAIL2WanModel", dict(WAN, in_dim=16), VID(), ctx(), {}),
 ("Wan Dancer", "comfy.ldm.wan.model_wandancer.WanDancerModel", dict(WAN, audio_inject_layers=[0], music_dim=8, music_heads=2, music_latent_dim=8, music_feature_dim=4), VID(), ctx(), {}),
 ("Cosmos Video", "comfy.ldm.cosmos.model.GeneralDIT", dict(max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=32, num_blocks=2, num_heads=2, crossattn_emb_channels=DIM, pos_emb_cls="rope3d", block_x_format="THWBD", use_adaln_lora=True, adaln_lora_dim=64, extra_per_block_abs_pos_emb=True, extra_per_block_abs_pos_emb_type="learnable"), VID(), ctx(), {}),
 ("Z-Image pixel space", "comfy.ldm.lumina.model.NextDiTPixelSpace", dict(patch_size=4, in_channels=3, dim=32, n_layers=2, n_refiner_layers=1, n_heads=2, cap_feat_dim=DIM, axes_dims=(4, 6, 6), axes_lens=(64, 64, 64), decoder_hidden_size=16, decoder_num_res_blocks=1), IMG(3), ctx(), {"num_tokens": TEXT}),
 ("PiD", "comfy.ldm.pixeldit.pid.PidNet", dict(in_channels=3, num_groups=2, hidden_size=32, pixel_hidden_size=8, pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=2, pixel_depth=1, patch_size=4, txt_embed_dim=DIM, txt_max_length=TEXT, lq_hidden_dim=16, lq_num_res_blocks=1), IMG(3), ctx(), {"lq_latent": torch.randn(1, 16, 4, 4), "degrade_sigma": torch.zeros(1)}),
 ("Chroma Radiance", "comfy.ldm.chroma_radiance.model.ChromaRadiance", dict(in_channels=3, out_channels=3, context_in_dim=DIM, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, patch_size=4, qkv_bias=True, in_dim=64, out_dim=32, hidden_dim=16, n_layers=2, txt_ids_dims=[], vec_in_dim=8, nerf_hidden_size=16, nerf_mlp_ratio=2, nerf_depth=1, nerf_max_freqs=4, nerf_tile_size=0, nerf_final_head_type="conv", nerf_embedder_dtype=None, use_x0=False, use_sequential_txt_ids=False), IMG(3), ctx(), {"guidance": torch.zeros(1)}),
 ("Stable Cascade C", "comfy.ldm.cascade.stage_c.StageC", dict(c_cond=32, c_hidden=[32, 32], nhead=[2, 2], blocks=[[1, 1], [1, 1]], block_repeat=[[1, 1], [1, 1]], c_clip_text=DIM, c_clip_text_pooled=8, c_clip_img=8, c_r=16), IMG(16), ctx(), {"__text_as": "clip_text", "clip_text_pooled": torch.randn(1, 1, 8), "clip_img": torch.randn(1, 1, 8)}),
 ("Stable Cascade B", "comfy.ldm.cascade.stage_b.StageB", dict(c_cond=32, c_hidden=[16, 32], nhead=[-1, 2], blocks=[[1, 1], [1, 1]], block_repeat=[[1, 1], [1, 1]], level_config=["CT", "CTA"], c_clip=DIM, c_r=16, c_effnet=4, dropout=[0, 0.0]), IMG(4), ctx(), {"__text_as": "clip", "effnet": torch.randn(1, 4, 16, 16)}),
 ("MiniMax H3", "comfy.ldm.minimax.model.MiniMaxH3Model", dict(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512, latents_dim=16, audio_latents_dim=8, text_dim=DIM, time_embed_hidden_size=256, time_embed_dim=128), VID(), ctx(), {"__second_stream": torch.randn(1, 8, 2, 16)}),
 ("ACE-Step 1.5", "comfy.ldm.ace.ace_step15.AceStepConditionGenerationModel", dict(in_channels=24, hidden_size=32, text_hidden_dim=DIM, timbre_hidden_dim=8, audio_acoustic_hidden_dim=8, num_dit_layers=2, num_lyric_layers=1, num_timbre_layers=1, num_tokenizer_layers=1, num_heads=2, num_kv_heads=2, head_dim=16, intermediate_size=64, fsq_dim=32, encoder_hidden_size=32, encoder_intermediate_size=64, encoder_num_heads=2, fsq_levels=[8, 8, 5]), torch.randn(1, 8, 16), ctx(), {"lyric_embed": torch.randn(1, 4, DIM), "refer_audio": torch.randn(1, 8, 16)}),
 ("Trellis 2", "comfy.ldm.trellis2.model.Trellis2", dict(resolution=16, in_channels=8, out_channels=8, model_channels=32, cond_channels=DIM, num_blocks=2, num_heads=2), torch.randn(1, 8, 16, 16, 16), ctx(), {"embeds": ctx()}),
 ("Hunyuan3D 2", "comfy.ldm.hunyuan3d.model.Hunyuan3Dv2", dict(in_channels=8, context_in_dim=DIM, hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=1, depth_single_blocks=1), torch.randn(1, 8, 16), ctx(), {}),
 ("Hunyuan3D 2.1", "comfy.ldm.hunyuan3dv2_1.hunyuandit.HunYuanDiTPlain", dict(in_channels=8, hidden_size=32, context_dim=DIM, depth=3, num_heads=2, num_moe_layers=1, guidance_cond_proj_dim=None, num_experts=2, moe_top_k=1), torch.randn(1, 8, 16), ctx(), {}),
 ("TripoSplat", "comfy.ldm.triposplat.model.LatentSeqMMFlowModel", dict(q_token_length=16, in_channels=8, model_channels=64, cond_channels=DIM, out_channels=8, num_blocks=2, num_refiner_blocks=1, num_head_channels=32), torch.randn(1, 16, 8), ctx(), {"__second_stream": torch.randn(1, 1, 5)}),
 ("YuE 2", "comfy.ldm.yue2.model.YuE2", dict(config={"hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 64}), torch.randn(1, 64, 16), ctx(2 * 2 * 2 * 128), {"yue2_chunks": [(0, 16, 0, TEXT)]}),
 ("MiniMax Music 3", "comfy.ldm.minimax_music.dit.MiniMaxMusic3DiT", dict(), torch.randn(1, 128, 16), ctx(8 * 4096, 32), {"conditioning_scale": torch.ones(1, 1, 1)}),
 ("SenseNova U1.5", "comfy.ldm.sensenova.model.SenseNovaU15", dict(), IMG(3, 64), ctx(4096), {"__set": ("comfy.ldm.sensenova.model", {"NUM_LAYERS": 1}), "text_input_ids": torch.randint(0, 1000, (1, 8))}),
 ("Anima", "comfy.ldm.anima.model.Anima", dict(max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=32, num_blocks=2, num_heads=2, crossattn_emb_channels=DIM, pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=16), VID(), ctx(), {}),
]
class Patcher:
    def __init__(self, core):
        self.model = type("Base", (), {})()
        self.model.diffusion_model = core
        self.model_options = {}
def load(path):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)
def build(path, kwargs):
    torch.manual_seed(1)
    cls = load(path)
    # A network with fixed sizes can still be built with fewer layers.
    for module, values in [kwargs.pop("__set")] if "__set" in kwargs else []:
        for key, value in values.items():
            setattr(importlib.import_module(module), key, value)
    sig = inspect.signature(cls.__init__).parameters
    args = dict(kwargs)
    if "operations" in sig or any(p.kind == p.VAR_KEYWORD for p in sig.values()):
        args.setdefault("operations", ops)
    for key in ("dtype", "device"):
        if key in sig: args.setdefault(key, torch.float32 if key == "dtype" else None)
    core = cls(**args)
    with torch.no_grad():
        for p in core.parameters():
            if p.is_floating_point(): p.normal_(0, 0.02)
        for b in core.buffers():
            if b.is_floating_point() and not torch.isfinite(b).all(): b.zero_()
    # ComfyUI's loaded weights do not require gradients, and some attention code insists on it.
    return core.requires_grad_(False).eval()
def forward_kwargs(core, x, context, extra):
    sig = inspect.signature(core.forward).parameters
    timestep = torch.tensor([0.5])
    t_name = next(n for n in ("timestep", "timesteps", "t", "r") if n in sig)
    return t_name, timestep, {**extra}
def call_for(core, x, context, extra, options):
    t_name, timestep, kw = forward_kwargs(core, x, context, extra)
    text_as = kw.pop("__text_as", "context")
    second = kw.pop("__second_stream", None)  # some models take two latents: ComfyUI unpacks them before the network
    def denoiser(input_x, t, c_crossattn=None, transformer_options=None, **rest):
        params = {t_name: t, text_as: c_crossattn, **rest}
        if second is not None:
            input_x = [input_x, second]
        if "transformer_options" in inspect.signature(core.forward).parameters or any(p.kind == p.VAR_KEYWORD for p in inspect.signature(core.forward).parameters.values()):
            params["transformer_options"] = transformer_options
        return core(input_x, **params)
    return denoiser, {"input": x, "timestep": timestep, "c": {"c_crossattn": context, "transformer_options": options, **kw}, "cond_or_uncond": [0]}
def key_order(core, x, context, extra):
    """Change only the text and see which key positions change, per attention call."""
    runs = []
    for text in (context, torch.randn_like(context)):
        keys = []
        def override(func, q, k, v, heads, *args, **kwargs):
            q_t, k_t = getattr(q, "tensor", q), getattr(k, "tensor", k)
            ql = int(q_t.shape[-2] if q_t.ndim == 4 else q_t.shape[1])
            keys.append((ql, k_t.detach().float().reshape(k_t.shape[0], k_t.shape[-2] if k_t.ndim == 4 else k_t.shape[1], -1)[0].clone() if k_t.ndim == 3 else k_t.detach().float()[0].transpose(0, 1).reshape(k_t.shape[-2], -1).clone()))
            return func(q, k, v, heads, *args, **kwargs)
        denoiser, args = call_for(core, x, text, extra, {"optimized_attention_override": override})
        torch.manual_seed(3)
        with torch.inference_mode():
            if type(core).__name__ in BYPASS_ATTENTION_MODELS:
                with reach_attention(core, override):
                    denoiser(args["input"], args["timestep"], **args["c"])
            else:
                denoiser(args["input"], args["timestep"], **args["c"])
        runs.append(keys)
    out = []
    n = context.shape[1]
    for (ql, a), (_, b) in zip(*runs):
        if a.shape != b.shape: continue
        change = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)
        changed = (change > 1e-4).nonzero().flatten().tolist()
        if changed and len(changed) < a.shape[0]:
            out.append(f"q{ql}/k{a.shape[0]}: text at {changed[0]}..{changed[-1]}" + ("" if changed == list(range(changed[0], changed[-1] + 1)) else " (gaps)"))
        elif changed and len(out) < 1:
            out.append(f"q{ql}/k{a.shape[0]}: all keys change")
        if len(out) >= 2: break
    return out or (["no calls reached the override"] if not runs[0] else ["text never changes keys"])
def text_position(found, text_start, tokens, observer, shape):
    """The map must look for text where changing the text really changed the keys."""
    for line in found:
        hit = re.match(r"q(\d+)/k(\d+): text at (\d+)\.\.(\d+)", line)
        if not hit: continue
        q, k, first, last = map(int, hit.groups())
        if k == tokens: return "cross"
        expected = k - tokens if text_start == -1 else text_start
        if observer._image_first:
            expected = observer._image_layout(shape, tokens)[0] or q
        return "matches" if first == expected else f"WRONG: map reads text at {expected}, model has it at {first}"
    return "not measured"
BLOCKS = ["audio:0", "audio:1", "audio:2", "double:0", "single:0", "input:1", "input:2", "input:4", "middle:0", "output:1", "output:2", "output:3"] + [f"call:{i}" for i in range(10)]
results = []
only = os.environ.get("ONLY")
for name, path, kwargs, x, context, extra in FAMILIES:
    if only and only.lower() not in name.lower(): continue
    if "__set" in extra:
        kwargs = {**kwargs, "__set": extra["__set"]}
        extra = {k: v for k, v in extra.items() if k != "__set"}
    row = {"family": name}
    try:
        core = build(path, kwargs)
    except Exception as exc:
        row.update(stage="construct", error=f"{type(exc).__name__}: {str(exc)[:170]}", tb_full=traceback.format_exc()[-900:] if os.environ.get("TB") else None); results.append(row); print(json.dumps(row), flush=True); continue
    try:
        denoiser, args = call_for(core, x, context, extra, {})
        with torch.inference_mode():
            plain = denoiser(args["input"], args["timestep"], **args["c"])
        plain = plain[0] if isinstance(plain, (tuple, list)) else plain
    except Exception as exc:
        row.update(stage="forward", error=f"{type(exc).__name__}: {str(exc)[:170]}", tb=traceback.format_exc().splitlines()[-3][:120], tb_full=traceback.format_exc()[-900:] if os.environ.get("TB") else None); results.append(row); print(json.dumps(row), flush=True); continue
    try:
        paths = all_layer_paths(core)
        row["layers"] = len(paths) - 1
        row["block_names"] = sorted({re.sub(r"\.\d+$", "", p) for p in paths if p})
        plan = all_layer_plan(paths, 2, 1024)
        published = []
        observer = DenoiserObserver(Patcher(core), paths, publish=lambda t: published.append(encode_trace(t)), **plan)
        options = {"optimized_attention_override": observer.attention_override}
        denoiser, args = call_for(core, x, context, extra, options)
        with torch.inference_mode():
            seen = observer(denoiser, args)
        seen = seen[0] if isinstance(seen, (tuple, list)) else seen
        row["unchanged"] = bool(torch.equal(torch.nan_to_num(plain), torch.nan_to_num(seen)))
        events = observer.trace["events"]
        row["events"] = len(events)
        row["shapes"] = sorted({str(e.get("shape")) for e in events if e.get("status") == "captured"})[:3]
        row["no_tensor"] = sum(e.get("status") != "captured" for e in events)
        row["views"] = sorted({(e.get("slices") or {}).get("layout") or "prefix" for e in events if e.get("status") == "captured"})
        row["published"] = len(published)
    except Exception as exc:
        row.update(stage="observe", error=f"{type(exc).__name__}: {str(exc)[:170]}", tb=traceback.format_exc().splitlines()[-3][:120]); results.append(row); print(json.dumps(row), flush=True); continue
    try:
        row["text_keys"] = key_order(core, x, context, extra)
    except Exception as exc:
        row["text_keys"] = [f"could not read: {type(exc).__name__}: {str(exc)[:80]}"]
    text_start = observer._text_start
    maps, reasons = [], []
    for block in BLOCKS:
        try:
            published = []
            observer = DenoiserObserver(Patcher(core), [""], max_calls=1, elements=8, publish=lambda t: published.append(encode_trace(t)), attention_block=block, token_labels=[f"w{i}" for i in range(context.shape[1])])
            options = {"optimized_attention_override": observer.attention_override}
            denoiser, args = call_for(core, x, context, extra, options)
            with torch.inference_mode():
                observer(denoiser, args)
            section = observer.trace.get("cross_attention") or {}
            if section.get("events") and published and os.environ.get("EXPORT_TRACE"):
                # A trace to import into the app and look at.
                with open(os.environ["EXPORT_TRACE"], "wb") as stream:
                    stream.write(published[-1])
            for ev in section.get("events", []):
                maps.append(f"{block}={ev['attention_kind']}:{ev['grid'][0]}x{ev['grid'][1]}x{ev.get('frames', 1)}")
            for s in section.get("skipped", []):
                reasons.append(s["reason"])
        except Exception as exc:
            reasons.append(f"RAISED {type(exc).__name__}: {str(exc)[:100]}")
    row["maps"] = maps[:5]
    # Only joint maps depend on this: a cross map's keys are all text, and a self-attention call can look alike.
    if any("=joint:" in m for m in maps):
        row["text_position"] = text_position(row["text_keys"], text_start, context.shape[1], observer, list(x.shape))
    row["map_reasons"] = sorted(set(re.sub(r"block [\w:]+", "block", r) for r in reasons))[:3] if not maps else []
    results.append(row)
    print(json.dumps(row), flush=True)
ok = [r for r in results if "error" not in r]
print("SUMMARY built", len(ok), "of", len(results), "; with maps", sum(bool(r.get("maps")) for r in ok), "; output unchanged", sum(r.get("unchanged") is True for r in ok), "; found layers", sum(r.get("layers", 0) > 0 for r in ok))
