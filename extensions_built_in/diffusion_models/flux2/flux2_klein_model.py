import os
import torch
from pathlib import Path

from .src.pipeline import Flux2Pipeline
from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_OFFLINE"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "0"

class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    # ── 1. 新增：对齐ZImage 顶层Buffer设备同步补丁 ──
    def _ensure_buffers_on_device(self, model: torch.nn.Module, device: torch.device):
        """
        Ensure all direct buffers and parameters on the model are on the correct device.
        This is needed because MemoryManager.to() only moves tracked submodules,
        not direct buffers like positional embeddings, guidance embeddings etc.
        """
        # 遍历模型直属buffer(不递归子模块)，移至目标设备
        for name, buf in model.named_buffers(recurse=False):
            if buf.device != device:
                setattr(model, name, buf.to(device))
        # 遍历模型直属参数(不递归子模块)，移至目标设备
        for name, param in model.named_parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device)

    # ── 2. 重写TE加载：对齐ZImage 分层卸载+量化执行顺序 ──
    def load_te(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen3")

        model_base_path = self.model_config.name_or_path
        local_te_path = os.path.join(model_base_path, "text_encoder")
        
        print(f"[调试] 文本编码器加载路径: {local_te_path}")
        print(f"[调试] 路径是否存在: {os.path.exists(local_te_path)}")
        if os.path.exists(local_te_path):
            print(f"[调试] 目录内文件列表: {os.listdir(local_te_path)}")

        if os.path.exists(local_te_path):
            te_load_path = local_te_path
        else:
            te_load_path = self.flux2_klein_te_path

        # 步骤1：加载模型权重（默认驻留CPU）
        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            te_load_path,
            torch_dtype=dtype,
        )

        # 步骤2：绑定分层显存卸载管理器（ZImage标准顺序：先绑定再移设备）
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        # 步骤3：移动到目标计算设备
        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        # 步骤4：文本编码器量化（对齐ZImage 使用独立qtype_te配置项）
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        tokenizer = Qwen2TokenizerFast.from_pretrained(te_load_path)
        return text_encoder, tokenizer

    # ── 3. 重写全量加载逻辑：完全对齐ZImage 加载流程 ──
    def load_model(self):
        dtype = self.torch_dtype
        model_path = self.model_config.name_or_path

        # 1. 加载 Transformer（复用父类逻辑，包含量化/分层卸载/低显存处理）
        self.print_and_status_update("Loading Flux2 Klein transformer")
        transformer_path = self._resolve_transformer_path(model_path)
        transformer = self._load_transformer(transformer_path, dtype)

        # 2. 加载 Text Encoder（使用本类重写的加载逻辑）
        text_encoder, tokenizer = self.load_te()

        # 3. 加载 VAE（复用父类自动检测逻辑）
        vae = self._load_vae(model_path, dtype)

        # 4. 初始化训练调度器
        self.noise_scheduler = self.get_train_scheduler()
        self.print_and_status_update("Making pipe (aligned with ZImage separate assignment)")

        # 对齐ZImage：先构造空流水线，量化场景下分开赋值效果更好
        pipe = Flux2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            text_encoder_type=self.flux2_te_type,
            is_guidance_distilled=self.flux2_is_guidance_distilled,
        )
        # 分步赋值组件，避免量化权重重复加载
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        # 5. 设备放置（严格对齐ZImage低显存策略）
        self.print_and_status_update("Preparing Model")
        # Transformer：非低显存模式移GPU，低显存模式留CPU
        if not self.model_config.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)
        
        # Text Encoder：始终常驻GPU，避免每次编码跨设备搬运
        pipe.text_encoder.to(self.device_torch)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
        flush()

        # 6. 赋值到实例变量
        self.vae = vae
        self.text_encoder = [pipe.text_encoder]
        self.tokenizer = [pipe.tokenizer]
        self.model = pipe.transformer
        self.pipeline = pipe

        self.print_and_status_update("Model Loaded")

    # ── 4. 重写训练前向：对齐ZImage 设备同步逻辑 ──
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings,
        guidance_embedding_scale: float,
        batch = None,
        **kwargs,
    ):
        # 对齐ZImage：前向前确保Transformer在GPU，且顶层Buffer设备正确
        self.model.to(self.device_torch)
        self._ensure_buffers_on_device(self.model, self.device_torch)

        # 复用父类完整前向计算逻辑
        return super().get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            guidance_embedding_scale=guidance_embedding_scale,
            batch=batch,
            **kwargs
        )

    # ── 5. 重写推理生成：对齐ZImage 设备同步逻辑 ──
    def generate_single_image(
        self,
        pipeline,
        gen_config,
        conditional_embeds,
        unconditional_embeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # 对齐ZImage：生成本步前确保模型在GPU，同步顶层Buffer设备
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        self._ensure_buffers_on_device(self.model, self.device_torch)

        # 复用父类完整生成逻辑
        return super().generate_single_image(
            pipeline=pipeline,
            gen_config=gen_config,
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            generator=generator,
            extra=extra,
        )


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"